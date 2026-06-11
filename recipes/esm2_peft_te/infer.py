# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from dataset import format_output_rows, load_input, write_output


def _batched_inference(
    model,
    tokenizer,
    records,
    batch_size: int,
    max_seq_length: int,
    stride: int,
    infer_overflowing_aas: bool,
    device: str = "cuda",
) -> tuple[list[str], list[int]]:
    id2label = model.config.id2label

    predictions = []
    sequences_to_sample_mapping = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        sequences = [r["sequence"] for r in batch]

        inputs = tokenizer(
            sequences,
            max_length=max_seq_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=infer_overflowing_aas,
            return_tensors="pt",
            padding=True,
        )

        num_samples = len(inputs["input_ids"])
        overflow_map = inputs.pop("overflow_to_sample_mapping", torch.arange(num_samples))

        # inner batching over tokenizer outputs
        for j in range(0, num_samples, batch_size):
            sub_inputs = {k: v[j : j + batch_size].to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**sub_inputs)

            preds = outputs.logits.argmax(dim=-1)

            for k, (pred, input_ids) in enumerate(zip(preds, sub_inputs["input_ids"])):
                length = (input_ids != tokenizer.pad_token_id).sum().item()
                labels = "".join(id2label[i.item()] for i in pred[:length])

                predictions.append(labels)

                # map back to original record index
                original_idx = i + overflow_map[j + k].item()
                sequences_to_sample_mapping.append(original_idx)

    return predictions, sequences_to_sample_mapping


@hydra.main(config_path="hydra_config", config_name="L0_sanity_infer", version_base="1.2")
def main(args: DictConfig):
    """Infer using a PEFT ESM-2 model.

    This script can be run once ESM2 has been PEFT fine-tuned and adapters have
    been checkpointed. For reference, an example has been provided in the './checkpoints' directory.
    """
    # Ideally we would like to load the PEFT model directly by doing:
    # >>> model = AutoPeftModelForTokenClassification.from_pretrained("<save_directory>", trust_remote_code=True)
    #
    # However, the from_pretrained() function has a positional argument named 'config' which prevent us from passing a
    # a different model config to the base_model. Thus, we first build the base model and then we load the PEFT adapters.

    # Load the custom config
    config = AutoConfig.from_pretrained(args.base_model_config_dir, trust_remote_code=True)

    # For recipe simplicity, we only support the attention input format to BSHD.
    config.attn_input_format = "bshd"

    # Load base model with the custom config
    base_model = AutoModelForTokenClassification.from_pretrained(
        args.model_tag,  # original model tag
        config=config,
        trust_remote_code=True,
    )

    # Load PEFT adapters on top
    peft_model = PeftModel.from_pretrained(base_model, args.peft_model_config_dir)
    peft_model = peft_model.to("cuda").eval()

    tokenizer = AutoTokenizer.from_pretrained("nvidia/esm2_t48_15B_UR50D")

    records = load_input(Path(args.input_file))

    predictions, sequences_to_sample_mapping = _batched_inference(
        peft_model,
        tokenizer,
        records,
        **args.inference,
    )

    if args.output_file:
        write_output(records, predictions, sequences_to_sample_mapping, Path(args.output_file))

    header, rows = format_output_rows(records, predictions, sequences_to_sample_mapping)

    print("---------------")
    print("\t".join(header))
    for row in rows:
        print("\t".join(row))


if __name__ == "__main__":
    main()
