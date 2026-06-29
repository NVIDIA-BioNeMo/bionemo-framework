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


# Add parent directory to sys.path so we can import from the model package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp
from transformers import AutoModelForCausalLM, AutoTokenizer

from collator import _split_batch_by_cp_rank
from convert import convert_llama_hf_to_te
from modeling_llama_te import NVLlamaForCausalLM


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

skip_in_ci = pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI.")

# TODO(@jomitchell): Delete once https://nvbugspro.nvidia.com/bug/5458694 is fixed.
requires_datacenter_hardware = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(
        gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
    ),
    reason="Test requires datacenter hardware (H100, H200, B100, B200, B300)",
)


def get_dummy_data_thd_with_padding(tokenizer):
    """Get dummy data for the THD format with padding for context parallelism.

    Args:
        tokenizer: The tokenizer to use.

    Returns:
        A dictionary containing the padded input ids, labels, and cu seq lens.
    """
    # Two English text sequences for testing with the llama3 model
    text1 = "The quick brown fox jumps over the lazy dog. This is a test sentence for context parallelism."
    text2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt."

    tok1 = tokenizer(text1, add_special_tokens=True)
    tok2 = tokenizer(text2, add_special_tokens=True)

    # Flatten inputs for THD format
    input_ids = torch.tensor(tok1["input_ids"] + tok2["input_ids"], dtype=torch.long)
    cu_seq_lens = torch.tensor(
        [0, len(tok1["input_ids"]), len(tok1["input_ids"]) + len(tok2["input_ids"])], dtype=torch.int32
    )

    # For causal LM, labels are the same as input_ids
    labels = input_ids.clone()

    # Pad sequences to be divisible by CP requirements (e.g., 32)
    pad_divisor = 32
    input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seq_lens,
        pad_divisor,
        padding_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        padding_label_id=-100,
    )

    # Calculate max sequence length
    seqlens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_length = int(seqlens.max().item())

    return {
        "input_ids": input_ids_padded.unsqueeze(0),
        "labels": labels_padded.unsqueeze(0),
        "cu_seq_lens_q": cu_seqlens_padded.to(torch.int32),
        "cu_seq_lens_k": cu_seqlens_padded.to(torch.int32),
        "cu_seq_lens_q_padded": cu_seqlens_padded.to(torch.int32),
        "cu_seq_lens_k_padded": cu_seqlens_padded.to(torch.int32),
        "max_length_q": max_length,
        "max_length_k": max_length,
    }


def get_te_model_checkpoint(tmp_path):
    """Get a TE model checkpoint for the Llama3 model.

    Args:
        tmp_path: The path to save the model checkpoint.

    Returns:
        The path to the saved model checkpoint.
    """
    # Use the 1B model for practical testing (8B model requires too much memory)
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16)
    model_te = convert_llama_hf_to_te(model_hf, attn_input_format="thd", self_attn_mask_type="padding_causal")
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"


def get_batch_for_cp_rank(batch, cp_rank, cp_world_size):
    """Get a batch for a given context parallelism rank.

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
    # Determine the max length of the sequence
    seqlens_q = batch_shard["cu_seq_lens_q_padded"][1:] - batch_shard["cu_seq_lens_q_padded"][:-1]
    batch_shard["max_length_q"] = int((seqlens_q.max().item() + 63) // 64 * 64)  # From TE code
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


@skip_in_ci
@requires_multi_gpu
@requires_datacenter_hardware
def test_context_parallel_equivalence_2process(recipe_path: Path, unused_tcp_port):
    """Test context parallel equivalence between 2 processes.

    In one instance, we run the model in non-distributed mode and in the other
    we run the model in distributed mode with context parallelism. We then compare
    the losses and logits from the two runs.

    We compare the (1) Losses, (2) Logits, and (3) Gradients from the two runs.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{unused_tcp_port}",
        os.path.relpath(__file__),
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=str(recipe_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_ckpt = get_te_model_checkpoint(tmp_path)

        # Create tokenizer for English text (use the 1B model tokenizer)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        input_data_thd_padded_dp0 = get_dummy_data_thd_with_padding(tokenizer)

        model = NVLlamaForCausalLM.from_pretrained(
            model_ckpt, attn_input_format="thd", self_attn_mask_type="padding_causal", torch_dtype=torch.bfloat16
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
            model.model.layers[0].self_attention.core_attention,
            model.model.layers[0].self_attention.layernorm_qkv,
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
        model = NVLlamaForCausalLM.from_pretrained(
            model_ckpt, attn_input_format="thd", self_attn_mask_type="padding_causal", torch_dtype=torch.bfloat16
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
        for i, transformer_layer in enumerate(model.module.model.layers):
            transformer_layer.set_context_parallel_group(
                cp_group, torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()), torch.cuda.Stream()
            )

        # Ensure model starts with clean slate
        model.zero_grad(set_to_none=True)

        # Create FRESH batch data for CP (don't reuse tensors from non-distributed run)
        batch = get_dummy_data_thd_with_padding(tokenizer=tokenizer)
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
            # For causal LM with CP, the per-rank loss computation has boundary issues because
            # the label shifting in CE loss doesn't account for discontinuities at shard boundaries.
            # The logits comparison (below) is the authoritative test for CP correctness.
            # We use relaxed tolerance here to verify the losses are in the same ballpark.
            torch.testing.assert_close(
                average_cp_loss.cpu(),
                loss_nondistributed_for_comparison,
                atol=0.5,  # Higher tolerance for causal LM due to shard boundary effects
                rtol=0.25,
            )

        # Gather the logits from all CP ranks
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
            model.module.model.layers[0].self_attention.core_attention,
            model.module.model.layers[0].self_attention.layernorm_qkv,
        ]

        gradients_cp = {}
        for i, layer in enumerate(sample_layers_cp):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    key = f"layer_{i}.{name}"
                    gradients_cp[key] = param.grad.detach().clone().cpu()

        # Now we compare the CP grads from rank 0 to the grads from the non-distributed run.
        # Note: Due to differences in loss computation at shard boundaries for causal LM,
        # the gradients may differ more than for MLM models.
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
