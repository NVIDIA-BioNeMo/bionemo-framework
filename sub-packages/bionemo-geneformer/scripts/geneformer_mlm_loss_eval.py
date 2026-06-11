# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import functools
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Type

import torch
import torch.distributed
import torch.utils
import torch.utils.data
from megatron.core.transformer.module import Float16Module
from nemo.utils import logging
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoModelForMaskedLM

from bionemo.core.data.load import load
from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import collate
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.testing import megatron_parallel_state_utils


class GeneformerHFAdapter(torch.nn.Module):
    """An adapter class for running the HF model against our subset of tokens."""

    def __init__(self, hf_path: str, my_token_dict: Dict[str, int], nv_tokenizer: GeneTokenizer):
        """An adapter that filters and re-orders tokens to match our tokenizer but with the original indices."""
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(hf_path)
        self.my_token_dict = deepcopy(my_token_dict)
        self.nv_tokenizer = deepcopy(nv_tokenizer)
        self.n_tokens_nv = len(self.nv_tokenizer.vocab)
        self.n_tokens_hf = len(my_token_dict)

        # nvidia tokenizer has [cls] and [pad] first along with some others that do not overlap. This mapper
        hf_ordered_nv_tokenizer = {
            self.nv_tokenizer.pad_token: my_token_dict["<pad>"],
            self.nv_tokenizer.mask_token: my_token_dict["<mask>"],
            self.nv_tokenizer.cls_token: my_token_dict["<cls>"],
            self.nv_tokenizer.sep_token: my_token_dict["<eos>"],  # name doesn't really matter here
        }
        tokens = list(my_token_dict.items())
        for k, t in tokens[:4]:
            assert k.startswith("<")

        missing_nv_tokens = []
        extra_tokens_not_covered = []
        for ens, idx in list(my_token_dict.items())[4:]:
            assert ens.startswith("ENSG")
            if ens in nv_tokenizer.vocab.keys():
                hf_ordered_nv_tokenizer[ens] = idx
            else:
                if idx < self.n_tokens_hf:
                    missing_nv_tokens.append(idx)
                else:
                    extra_tokens_not_covered.append(idx)
        self.hf_ordered_nv_tokenizer = hf_ordered_nv_tokenizer
        self.extra_tokens_not_covered = extra_tokens_not_covered
        self.register_buffer("missing_nv_tokens", torch.tensor(missing_nv_tokens, dtype=int))

    @property
    def device(self) -> torch.device:
        """Return the device of this model."""
        # This is populated through the self.register_buffer call in init.
        return self.missing_nv_tokens.device

    def get_tokenizer(self) -> GeneTokenizer:
        """Return the filtered tokenizer with keys that match the order of the nv model."""
        nv_tok = deepcopy(self.nv_tokenizer)
        # HF tokenizer only has pad and mask, no other special tokens.
        nv_tok.special_tokens = (nv_tok.mask_token, nv_tok.pad_token)  # type: ignore
        nv_tok.vocab = self.hf_ordered_nv_tokenizer
        nv_tok.decode_vocab = {v: k for k, v in nv_tok.vocab.items()}
        return nv_tok

    def forward(self, *args, **kwargs):
        """Run forward and return the logits."""
        logits = self.model(*args, **kwargs).logits
        # logits[:, :, self.missing_nv_tokens] = -torch.inf
        # breakpoint()
        return logits


def main(
    model_path: Path | None,
    hf_model_path: str,
    dataset_path: Path,
    hf_token_dictionary_path: Path,
    hf_medians_dictionary_path: Path,
    mask_prob: float = 0.15,
    batch_size: int = 16,
    precision: str = "bf16-mixed",
    config_class: Type[BioBertConfig] = GeneformerConfig,
    seq_len_nv: int = 2048,
    seq_len_hf: int = 2048,
    seed: int = 513,
    include_unrecognized_vocab_in_dataset: bool = False,
):
    """Inference function (requires DDP and only training data that fits in memory)."""
    # This is just used to get the tokenizer :(
    train_data_path: Path = (
        load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data" / "train"
    )
    n_devices: int = torch.cuda.device_count()
    assert n_devices > 0
    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            logging.info("*************** Preprocessing Finished ************")
        case _:
            logging.error("Failed to download the tokenizer for the NV geneformer model.")
            assert False
    with open(hf_token_dictionary_path, "rb") as geneformer_hf_token_file:
        geneformer_hf_token_dict = pickle.load(geneformer_hf_token_file)
    with open(hf_medians_dictionary_path, "rb") as geneformer_hf_median_file:
        geneformer_hf_medians_dict = pickle.load(geneformer_hf_median_file)
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        geneformer_nv_inferer_cfg = config_class(
            seq_length=seq_len_nv,
            params_dtype=get_autocast_dtype(precision),
            pipeline_dtype=get_autocast_dtype(precision),
            autocast_dtype=get_autocast_dtype(precision),  # setting this speeds things up a lot
            # handle checkpoint resumption here rather than auto-resume so this supports fine-tuning capabilities
            initial_ckpt_path=str(model_path) if model_path is not None else None,
            initial_ckpt_skip_keys_with_these_prefixes=[],  # load everything from the checkpoint.
        )
        geneformer_nv_inferer = Float16Module(
            geneformer_nv_inferer_cfg, geneformer_nv_inferer_cfg.configure_model(tokenizer).cuda(0 % n_devices)
        ).eval()

        # TODO only predict with tokens that exist in both models.

        hf_model = GeneformerHFAdapter(hf_model_path, geneformer_hf_token_dict, tokenizer).eval().cuda(1 % n_devices)
        hf_total_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
        nv_total_params = sum(p.numel() for p in geneformer_nv_inferer.parameters() if p.requires_grad)
        print(f"HF Model Params: {hf_total_params}, NV Model Params: {nv_total_params}", file=sys.stdout)
        tokenizer_filt = deepcopy(tokenizer)
        ori_nv_vocab_size: int = len(tokenizer.vocab)
        hf_tokenizer = hf_model.get_tokenizer()
        tokenizer_filt.vocab = {
            k: v for k, v in tokenizer.vocab.items() if k in hf_tokenizer.vocab or k in tokenizer.special_tokens
        }

        ds_nv = SingleCellDataset(
            dataset_path,
            tokenizer=tokenizer_filt,  # TODO replace with the filtered one.
            median_dict=median_dict,
            max_len=seq_len_nv,
            mask_prob=mask_prob,
            seed=seed,
            include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
        )
        ds_hf_nvfilt = SingleCellDataset(
            dataset_path,
            hf_tokenizer,
            geneformer_hf_medians_dict,
            max_len=seq_len_hf,
            mask_prob=mask_prob,
            eos_token=hf_tokenizer.token_to_id(hf_tokenizer.sep_token),  # Stored in the special token
            seed=seed,
            include_unrecognized_vocab_in_dataset=include_unrecognized_vocab_in_dataset,
        )
        print(f"Loaded dataset of length (NV): {len(ds_nv)}, (HF): {len(ds_hf_nvfilt)}")

        dl_hf = DataLoader(
            ds_hf_nvfilt,
            batch_size=batch_size,
            sampler=[EpochIndex(epoch=0, idx=i) for i in range(len(ds_hf_nvfilt))],
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=ds_hf_nvfilt.tokenizer.pad_id,
                min_length=seq_len_hf,
                max_length=seq_len_hf,
            ),
        )
        dl_nv = DataLoader(
            ds_nv,
            batch_size=batch_size,
            sampler=[EpochIndex(epoch=0, idx=i) for i in range(len(ds_nv))],
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=ds_nv.tokenizer.pad_id,
                min_length=seq_len_nv,
                max_length=seq_len_nv,
            ),
        )

        with torch.no_grad():
            dl_hf_iter = iter(dl_hf)
            dl_nv_iter = iter(dl_nv)
            loss_hf = 0.0
            n_hf = 0
            loss_nv = 0.0
            n_nv = 0
            nv_device = geneformer_nv_inferer.module.embedding.position_embeddings.weight.device
            hf_device = hf_model.device
            for _ in trange(len(dl_hf)):
                batch_hf = {k: v.to(hf_device) for k, v in next(dl_hf_iter).items()}
                batch_nv = {k: v.to(nv_device) for k, v in next(dl_nv_iter).items()}
                logits_hf = hf_model(batch_hf["text"].long(), batch_hf["attention_mask"])
                loss_hf += (
                    torch.nn.functional.cross_entropy(
                        logits_hf[batch_hf["loss_mask"]],
                        batch_hf["labels"][batch_hf["loss_mask"]],
                        reduction="sum",
                    )
                    .cpu()
                    .sum()
                    .item()
                )
                n_hf += batch_hf["loss_mask"].sum().cpu().item()

                logits_nv = (
                    geneformer_nv_inferer(batch_nv["text"], batch_nv["attention_mask"])["token_logits"]
                    .transpose(0, 1)
                    .contiguous()
                )
                loss_nv += (
                    torch.nn.functional.cross_entropy(
                        logits_nv[batch_nv["loss_mask"]][..., :ori_nv_vocab_size],
                        batch_nv["labels"][batch_nv["loss_mask"]],
                        reduction="sum",
                    )
                    .cpu()
                    .sum()
                    .item()
                )
                n_nv += batch_nv["loss_mask"].sum().cpu().item()
        print(f"NV mean loss: {loss_nv / n_nv}")
        print(f"HF mean loss: {loss_hf / n_hf}")


def entrypoint():
    """Main entry point for running the evaluation."""
    parser = argparse.ArgumentParser(description="MLM Performance vs HF Script")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to nvidia geneformer model checkpoint (unless you want random weights)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--hf-token-dictionary-path",
        type=Path,
        help="Path to token dictionary file. "
        "Eg `wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary_gc95M.pkl`"
        "then provide the path to the downloaded file.",
        required=True,
    )
    parser.add_argument(
        "--hf-medians-dictionary-path",
        type=Path,
        help="Path to token dictionary file. "
        "Eg `wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_median_dictionary_gc95M.pkl` "
        "then provide the path to the downloaded file.",
        required=True,
    )
    parser.add_argument("--hf-model-path", type=str, default="ctheodoris/Geneformer", help="HF model path")
    parser.add_argument("--dataset-path", type=Path, help="Path to dataset directory", required=True)
    parser.add_argument(
        "--include-unrecognized-vocab-in-dataset",
        action="store_true",
        help="If set to true, a hard-check is performed to verify all gene identifers are in the user supplied tokenizer vocab. Defaults to false which means any gene identifier not in the user supplied tokenizer vocab will be excluded.",
    )

    args = parser.parse_args()
    main(
        args.model_path,
        args.hf_model_path,
        args.dataset_path,
        args.hf_token_dictionary_path,
        args.hf_medians_dictionary_path,
        args.include_unrecognized_vocab_in_dataset,
    )


if __name__ == "__main__":
    entrypoint()
