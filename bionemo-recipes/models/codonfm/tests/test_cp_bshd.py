# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Context parallel equivalence test for CodonFM in BSHD format.

Verifies that running the model with context parallelism (CP=2) produces
equivalent losses, logits, and gradients compared to a non-distributed run.
"""

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
from collator import _split_batch_by_cp_rank
from dataset import CodonMLMCollator
from modeling_codonfm_te import MODEL_PRESETS, CodonFMConfig, CodonFMForMaskedLM
from tokenizer import CodonTokenizer
from torch.distributed.device_mesh import init_device_mesh


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


# Test codon sequences (DNA strings of 3-mer codons) - chosen to produce 32-token sequences with special tokens
TEST_CODON_SEQUENCES = [
    "ATGCGTAAAGCTGTTCAGGATCTGAATGCCATCTATGCGATGCGTAAAGCTGTTCAGGATCTGAATGCCATCTATGCG",
    "ATGGATCGTACCGCTGAACAGCGTCTGATCAAAGCCATGGATCGTACCGCTGAACAGCGTCTGATCAAAGCCATGGAT",
]


def get_dummy_data_bshd_with_padding():
    """Create dummy BSHD data with padding for CP.

    Returns:
        A dictionary containing the padded input ids and labels in BSHD format [batch, seq_len].
    """
    tokenizer = CodonTokenizer()
    collator = CodonMLMCollator(
        tokenizer=tokenizer,
        max_seq_length=32,  # Pad to 32 for CP divisibility (32 / (2*2) = 8)
        mlm_probability=0.0,  # No masking for deterministic testing
    )
    samples = [{"sequence": seq} for seq in TEST_CODON_SEQUENCES]
    batch = collator(samples)
    batch["labels"] = batch["input_ids"].clone()  # Identity for testing CP sanity.
    del batch["attention_mask"]
    return batch


def create_model_checkpoint(tmp_path):
    """Create a small CodonFM model checkpoint.

    Args:
        tmp_path: The path to save the model checkpoint.

    Returns:
        The path to the saved model checkpoint.
    """
    config = CodonFMConfig(
        attn_input_format="bshd",
        dtype=torch.bfloat16,
        **MODEL_PRESETS["encodon_200k"],
    )
    model = CodonFMForMaskedLM(config)
    model.save_pretrained(tmp_path / "codonfm_checkpoint")
    return tmp_path / "codonfm_checkpoint"


def get_batch_for_cp_rank(batch, cp_rank, cp_world_size):
    """Get a batch shard for a given context parallelism rank.

    Args:
        batch: The batch to get a shard of.
        cp_rank: The context parallelism rank.
        cp_world_size: The size of the context parallelism group.

    Returns:
        A dictionary containing the shard of the batch.
    """
    input_ids_sharded, labels_sharded = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
    )
    batch_shard = dict(batch)
    batch_shard["input_ids"] = input_ids_sharded
    batch_shard["labels"] = labels_sharded
    return batch_shard


@dataclass(frozen=True)
class DistributedConfig:
    """Class to track distributed ranks and handle basic distributed training setup."""

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))
    _master_addr: str = field(default_factory=lambda: os.environ.setdefault("MASTER_ADDR", "localhost"))
    _master_port: str = field(default_factory=lambda: os.environ.setdefault("MASTER_PORT", "12355"))

    def is_main_process(self) -> bool:
        """This is the global rank 0 process."""
        return self.rank == 0


def test_context_parallel_equivalence_1process():
    """Smoke test to ensure CP works with 1 process and matches non-distributed results."""
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
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


@requires_multi_gpu
def test_context_parallel_equivalence_2process():
    """Test CP equivalence between 2 processes.

    Compares (1) Losses, (2) Logits, and (3) Gradients from distributed CP vs non-distributed runs.
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
        model_ckpt = create_model_checkpoint(tmp_path)

        input_data_bshd_padded = get_dummy_data_bshd_with_padding()

        config = CodonFMConfig.from_pretrained(model_ckpt)
        config.attn_input_format = "bshd"
        config.dtype = torch.bfloat16
        model = CodonFMForMaskedLM(config)
        model.load_state_dict(torch.load(model_ckpt / "model.safetensors", weights_only=True), strict=False)
        model = model.to(dtype=torch.bfloat16, device="cuda")
        model.train()

        input_data_bshd_padded = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_data_bshd_padded.items()
        }
        outputs_nondistributed = model(**input_data_bshd_padded)
        loss_nondistributed = outputs_nondistributed.loss
        loss_nondistributed.backward()

        # Clone everything we need for later comparison BEFORE deleting
        loss_nondistributed_for_comparison = loss_nondistributed.detach().clone().cpu()
        logits_nondistributed_for_comparison = outputs_nondistributed.logits.detach().clone().cpu()

        # Sample gradients from a few layers for comparison
        sample_layers = [
            model.encoder.layers[0].self_attention.core_attention,
            model.encoder.layers[0].self_attention.layernorm_qkv,
        ]

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
        del model, outputs_nondistributed, loss_nondistributed, input_data_bshd_padded
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Initialize distributed training
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        torch.cuda.set_device(dist_config.local_rank)

        ddp_size = 1
        cp_size = torch.distributed.get_world_size()
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(ddp_size, cp_size),
            mesh_dim_names=("ddp", "cp"),
        )

        # Re-initialize the model on the new device
        config = CodonFMConfig.from_pretrained(model_ckpt)
        config.attn_input_format = "bshd"
        config.dtype = torch.bfloat16
        model = CodonFMForMaskedLM(config)
        model.load_state_dict(torch.load(model_ckpt / "model.safetensors", weights_only=True), strict=False)
        model = model.to(dtype=torch.bfloat16, device=device)
        model.train()
        model.zero_grad(set_to_none=True)

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
        for transformer_layer in model.module.encoder.layers:
            transformer_layer.set_context_parallel_group(
                cp_group, torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()), torch.cuda.Stream()
            )

        model.zero_grad(set_to_none=True)

        # Create FRESH batch data for CP
        batch = get_dummy_data_bshd_with_padding()
        batch = {k: v.detach().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_cp = get_batch_for_cp_rank(batch, cp_rank=cp_rank, cp_world_size=cp_world_size)
        batch_cp["max_length_q"] = batch_cp["max_length_k"] = 32

        torch.distributed.barrier(group=cp_group)

        outputs_cp = model(**batch_cp)

        # Gather losses from all cp ranks
        losses_list = [torch.zeros_like(outputs_cp.loss) for _ in range(cp_world_size)]
        torch.distributed.all_gather(losses_list, outputs_cp.loss, group=cp_group)

        if cp_rank == 0:
            average_cp_loss = torch.mean(torch.stack(losses_list))
            torch.testing.assert_close(
                average_cp_loss.cpu(),
                loss_nondistributed_for_comparison,
                atol=0.1,
                rtol=0.05,
            )

        # Gather logits from all CP ranks
        logits_contiguous = outputs_cp.logits.contiguous()
        logits_list = [torch.zeros_like(logits_contiguous) for _ in range(cp_world_size)]
        torch.distributed.all_gather(logits_list, logits_contiguous, group=cp_group)

        if cp_rank == 0:
            # Reconstruct the full logits from CP-split chunks for BSHD format
            batch_size, seq_len_sharded, vocab_size = logits_list[0].shape
            seq_len_full = batch["input_ids"].shape[1]
            total_chunks = 2 * cp_world_size
            chunk_size = seq_len_full // total_chunks

            reconstructed_logits = torch.zeros(
                (batch_size, seq_len_full, vocab_size), dtype=torch.bfloat16, device=logits_list[0].device
            )

            for batch_idx in range(batch_size):
                for cp_idx, logits_shard in enumerate(logits_list):
                    chunk_indices = [cp_idx, total_chunks - cp_idx - 1]
                    for chunk_pos, chunk_idx in enumerate(chunk_indices):
                        start_idx = chunk_idx * chunk_size
                        end_idx = start_idx + chunk_size
                        shard_start = chunk_pos * chunk_size
                        shard_end = shard_start + chunk_size
                        reconstructed_logits[batch_idx, start_idx:end_idx, :] = logits_shard[
                            batch_idx, shard_start:shard_end, :
                        ]

            assert reconstructed_logits.shape == logits_nondistributed_for_comparison.shape
            torch.testing.assert_close(
                reconstructed_logits.cpu(),
                logits_nondistributed_for_comparison,
                atol=0.29,
                rtol=0.01,
            )

        # Test gradient synchronization with DDP
        outputs_cp.loss.backward()

        sample_layers_cp = [
            model.module.encoder.layers[0].self_attention.core_attention,
            model.module.encoder.layers[0].self_attention.layernorm_qkv,
        ]

        gradients_cp = {}
        for i, layer in enumerate(sample_layers_cp):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    key = f"layer_{i}.{name}"
                    gradients_cp[key] = param.grad.detach().clone().cpu()

        if cp_rank == 0:
            for key in gradients_nondistributed.keys():
                if key in gradients_cp:
                    grad_cp = gradients_cp[key]
                    grad_nondist = gradients_nondistributed[key]

                    torch.testing.assert_close(
                        grad_cp,
                        grad_nondist,
                        atol=2e-3,
                        rtol=1e-2,
                        msg=lambda x: f"Gradients don't match for {key}: {x}",
                    )

        torch.distributed.destroy_process_group()
