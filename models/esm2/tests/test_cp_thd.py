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

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# When launched via torchrun, conftest.py sys.path setup doesn't run.
# Ensure the model directory (parent of tests/) is on sys.path for bare module imports.
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from collator import DataCollatorWithFlattening, _split_batch_by_cp_rank
from convert import convert_esm_hf_to_te
from modeling_esm_te import NVEsmForMaskedLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

# TODO(@jomitchell): Delete once https://nvbugspro.nvidia.com/bug/5458694 is fixed.
requires_datacenter_hardware = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(
        gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
    ),
    reason="Test requires datacenter hardware (H100, H200, B100, B200, B300)",
)


def get_dummy_data_thd_with_padding_dp0(tokenizer):
    """
    Get dummy data for the THD format with padding for context parallelism.
    Args:
        cp_size: The size of the context parallelism group.
        tokenizer: The tokenizer to use.
    Returns:
        A dictionary containing the padded input ids, labels, and cu seq lens.
    """
    # Two real protein sequences (30 amino acids each, will be 32 tokens with BOS/EOS)
    protein1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLL"  # 32 tokens with BOS/EOS
    protein2 = "MSHHWGYGKHNGPEHWHKDFPIAKGERFLL"  # 32 tokens with BOS/EOS

    tok1 = tokenizer(protein1, add_special_tokens=True)
    tok2 = tokenizer(protein2, add_special_tokens=True)

    data_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.0,
            pad_to_multiple_of=32,
            seed=42,
        ),
        pad_sequences_to_be_divisible_by=32,
    )
    batch = data_collator([tok1, tok2])
    batch["labels"] = batch["input_ids"].clone()  # We just use the identity function for testing CP sanity.
    return batch


def get_te_model_checkpoint(tmp_path):
    """
    Get a TE model checkpoint for the ESM2 model.
    Args:
        tmp_path: The path to save the model checkpoint.
    Returns:
        The path to the saved model checkpoint.
    """
    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"


def get_batch_for_cp_rank(batch, cp_rank, cp_world_size):
    """
    Get a batch for a given context parallelism rank.

    Args:
        batch: The batch to get a shard of.
        cp_rank: The context parallelism rank.
        cp_world_size: The size of the context parallelism group.
    Returns:
        A dictionary containing the shard of the batch.
    """
    input_ids_sharded, labels_sharded = _split_batch_by_cp_rank(
        cu_seqlens_padded=batch["cu_seq_lens_q_padded"],
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="thd",
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
    )
    batch_shard = dict(batch)
    batch_shard["input_ids"] = input_ids_sharded
    batch_shard["labels"] = labels_sharded
    # Now determine the max length of the sequence.
    seqlens_q = batch_shard["cu_seq_lens_q_padded"][1:] - batch_shard["cu_seq_lens_q_padded"][:-1]
    batch_shard["max_length_q"] = int((seqlens_q.max().item() + 63) // 64 * 64)  # From TE code.
    batch_shard["max_length_k"] = batch_shard["max_length_q"]
    return batch_shard


@dataclass(frozen=True)
class DistributedConfig:
    """Class to track distributed ranks and handle basic distributed training setup.

    If torch distributed environment variables are not set, we set them to default values for single-process training.

    Attributes:
        rank: The rank of the process.
        local_rank: The local rank of the process.
        world_size: The total number of processes.
    """

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))
    _master_addr: str = field(default_factory=lambda: os.environ.setdefault("MASTER_ADDR", "localhost"))
    _master_port: str = field(default_factory=lambda: os.environ.setdefault("MASTER_PORT", "12355"))

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0


@requires_multi_gpu
@requires_datacenter_hardware
def test_context_parallel_equivalence_2process():
    """
    Test the context parallel equivalence between 2 processes. In one instance, we run the model in non-distributed mode and in the other
    we run the model in distributed mode with context parallelism. We then compare the losses and logits from the two runs.

    We compare the (1) Losses, (2) Logits, and (3) Gradients from the two runs.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_ckpt = get_te_model_checkpoint(tmp_path)

        # Create tokenizer for real protein sequences
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f")
        input_data_thd_padded_dp0 = get_dummy_data_thd_with_padding_dp0(tokenizer)

        model = NVEsmForMaskedLM.from_pretrained(
            model_ckpt, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16
        )
        model.to("cuda")
        input_data_thd_padded_dp0 = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_thd_padded_dp0.items()
        }
        outputs_nondistributed = model(**input_data_thd_padded_dp0)
        loss_nondistributed = outputs_nondistributed.loss
        loss_nondistributed.backward()

        # Clone everything we need for later comparison BEFORE deleting
        loss_nondistributed_for_comparison = loss_nondistributed.detach().clone().cpu()
        logits_nondistributed_for_comparison = outputs_nondistributed.logits.detach().clone().cpu()

        # Sample gradients from a few layers for comparison
        sample_layers = [
            model.model.encoder.layers[0].self_attention.core_attention,
            model.model.encoder.layers[0].self_attention.layernorm_qkv,
        ]

        # Now grab the gradients from the sample layers
        gradients_nondistributed = {}
        for i, layer in enumerate(sample_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    key = f"layer_{i}.{name}"
                    gradients_nondistributed[key] = param.grad.detach().clone().cpu()

        # Now setup distributed training for CP.
        dist_config = DistributedConfig()
        device = torch.device(f"cuda:{dist_config.local_rank}")

        # Clean up everything from non-distributed run
        del model, outputs_nondistributed, loss_nondistributed, input_data_thd_padded_dp0
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        # Initialize distributed training
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        torch.cuda.set_device(dist_config.local_rank)
        # Create a device mesh for DDP=1, CP=2

        ddp_size = 1
        cp_size = 2
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(ddp_size, cp_size),
            mesh_dim_names=("ddp", "cp"),
        )
        # Re-initialize the model on the new device (fresh instance, no shared graph)
        model = NVEsmForMaskedLM.from_pretrained(
            model_ckpt, attn_input_format="thd", token_dropout=False, dtype=torch.bfloat16
        )
        model = model.to(device=device)
        model.train()  # Set to training mode to enable gradient computation
        model.zero_grad(set_to_none=True)  # Ensure no gradients from initialization

        group_fsdp_cp = device_mesh[("ddp", "cp")]._flatten("dp_cp").get_group()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank],
            output_device=dist_config.local_rank,
            process_group=group_fsdp_cp,
        )
        cp_group = device_mesh["cp"].get_group()
        cp_rank = device_mesh.get_local_rank("cp")
        cp_world_size = torch.distributed.get_world_size(group=cp_group)

        # Set up context parallelism for each layer
        for i, transformer_layer in enumerate(model.module.model.encoder.layers):
            transformer_layer.set_context_parallel_group(
                cp_group, torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()), torch.cuda.Stream()
            )

        # Ensure model starts with clean slate
        model.zero_grad(set_to_none=True)

        # Create FRESH batch data for CP (don't reuse tensors from non-distributed run)
        batch = get_dummy_data_thd_with_padding_dp0(tokenizer=tokenizer)
        # Move batch to CUDA and ensure tensors are detached from any previous graphs
        batch = {k: v.detach().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_cp = get_batch_for_cp_rank(batch, cp_rank=cp_rank, cp_world_size=cp_world_size)

        torch.distributed.barrier(group=cp_group)

        outputs_cp = model(**batch_cp)
        loss_cp = outputs_cp.loss

        # Gather the losses from all cp ranks (collective operation - all ranks must participate)
        losses_list = [torch.zeros_like(outputs_cp.loss) for _ in range(cp_world_size)]
        torch.distributed.all_gather(losses_list, outputs_cp.loss, group=cp_group)

        if cp_rank == 0:
            average_cp_loss = torch.mean(torch.stack(losses_list))
            # The average of per-rank losses should be close to the non-distributed loss
            # Note: They may not be exactly equal due to how loss is computed on sharded data
            torch.testing.assert_close(
                average_cp_loss.cpu(),
                loss_nondistributed_for_comparison,
                atol=0.1,  # Allow some difference due to loss computation on shards
                rtol=0.05,
            )

        # Gather the logits from all CP ranks
        # The logits are split along the sequence dimension (dim=1 for THD format: [batch, seq, vocab])
        logits_contiguous = outputs_cp.logits.contiguous()
        logits_list = [torch.zeros_like(logits_contiguous) for _ in range(cp_world_size)]
        torch.distributed.all_gather(logits_list, logits_contiguous, group=cp_group)

        if cp_rank == 0:
            # Reconstruct the full logits from CP-split chunks dynamically
            # Get sequence lengths from cu_seqlens
            cu_seqlens = batch["cu_seq_lens_q_padded"].cpu()
            num_seqs = len(cu_seqlens) - 1
            total_tokens = int(cu_seqlens[-1].item())
            vocab_size = logits_nondistributed_for_comparison.shape[-1]

            reconstructed_logits = torch.zeros((total_tokens, vocab_size), dtype=torch.bfloat16)

            # For each sequence, reconstruct from CP chunks
            cp_offset_rank0 = 0
            cp_offset_rank1 = 0

            for seq_idx in range(num_seqs):
                seq_start = int(cu_seqlens[seq_idx].item())
                seq_end = int(cu_seqlens[seq_idx + 1].item())
                seq_len = seq_end - seq_start
                chunk_size = seq_len // (2 * cp_world_size)  # Each sequence split into 2*cp_world_size chunks

                # CP rank 0 gets chunks [0, 3], CP rank 1 gets chunks [1, 2]
                for chunk_idx in range(2 * cp_world_size):
                    chunk_start_in_seq = seq_start + chunk_idx * chunk_size
                    chunk_end_in_seq = chunk_start_in_seq + chunk_size

                    if chunk_idx == 0 or chunk_idx == 3:  # Chunks for CP rank 0
                        reconstructed_logits[chunk_start_in_seq:chunk_end_in_seq, :] = logits_list[0][
                            cp_offset_rank0 : cp_offset_rank0 + chunk_size, :
                        ]
                        cp_offset_rank0 += chunk_size
                    else:  # Chunks 1, 2 for CP rank 1
                        reconstructed_logits[chunk_start_in_seq:chunk_end_in_seq, :] = logits_list[1][
                            cp_offset_rank1 : cp_offset_rank1 + chunk_size, :
                        ]
                        cp_offset_rank1 += chunk_size

            assert reconstructed_logits.shape == logits_nondistributed_for_comparison.shape
            cosine_sim = torch.nn.functional.cosine_similarity(
                reconstructed_logits.flatten().float().cuda(),
                logits_nondistributed_for_comparison.flatten().float().cuda(),
                dim=0,
            )
            assert cosine_sim > 0.99, f"Logits cosine similarity too low: {cosine_sim}"

        # Test gradient synchronization with DDP
        loss_cp = outputs_cp.loss
        loss_cp.backward()  # DDP automatically synchronizes gradients here

        # Capture gradients from the same layers in the CP model
        # Note: DDP wraps the model with 'module.' prefix
        sample_layers_cp = [
            model.module.model.encoder.layers[0].self_attention.core_attention,
            model.module.model.encoder.layers[0].self_attention.layernorm_qkv,
        ]

        gradients_cp = {}
        for i, layer in enumerate(sample_layers_cp):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    key = f"layer_{i}.{name}"
                    gradients_cp[key] = param.grad.detach().clone().cpu()

        # Now we compare the CP grads from rank 0 to the Grads from the non-distributed run. (they should be the same on each process for non dist)
        if cp_rank == 0:
            # Compare gradients between non-distributed and CP
            for key in gradients_nondistributed.keys():
                if key in gradients_cp:
                    grad_cp = gradients_cp[key]
                    grad_nondist = gradients_nondistributed[key]

                    cosine_sim = torch.nn.functional.cosine_similarity(
                        grad_cp.flatten().float(), grad_nondist.flatten().float(), dim=0
                    )
                    # These aren't as close, since the losses on the sharded CP ranks will be different.
                    assert cosine_sim > 0.8, f"Gradient cosine similarity too low for {key}: {cosine_sim}"

        torch.distributed.destroy_process_group()
