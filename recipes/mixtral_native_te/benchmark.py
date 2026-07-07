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

"""Standalone torchrun benchmark for Mixtral EP/FSDP2/MXFP8 throughput and MFU."""

from __future__ import annotations

import argparse
import os
import time


# Must be set before Transformer Engine grouped-linear / CuteDSL imports.
os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

import torch
import torch.distributed as dist
import transformer_engine.pytorch
from distributed_config import DistributedConfig
from distributed_setup import all_reduce_dense_grads_over_ep, build_mesh_and_wrap
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM, _ensure_fused_grouped_mlp_registered
from train_fsdp2_ep import _clip_grad_norm_mixed
from transformer_engine.common.recipe import Format, MXFP8BlockScaling
from transformer_engine.pytorch.optimizers import FusedAdam


# Dense peak FLOP/s per B200 GPU (datasheet, no sparsity credit).
PEAK_FLOPS_BF16 = 2.25e15
PEAK_FLOPS_FP8 = 4.5e15

DEFAULT_CONFIG = "./model_configs/mistralai/Mixtral-8x7B-v0.1"


def _assert_fused_mxfp8_supported() -> None:
    from transformer_engine.pytorch.ops.fused.forward_grouped_mlp import (
        ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8,
    )

    ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported.cache_clear()
    if not ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported():
        raise RuntimeError("ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8 is not supported on this hardware.")
    _ensure_fused_grouped_mlp_registered()


def compute_n_active(config: NVMixtralConfig) -> int:
    """Per-token active parameter count for MoE FLOP estimate (excludes embeddings/lm_head).

    Summed across all layers: each token activates one attention block, one router,
    and ``num_experts_per_tok`` expert FFNs per layer.
    """
    h = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = h // num_heads

    # Q/K/V/O projections (GQA), per layer.
    attn_params = (
        h * (num_heads * head_dim)  # q_proj
        + h * (num_kv_heads * head_dim)  # k_proj
        + h * (num_kv_heads * head_dim)  # v_proj
        + (num_heads * head_dim) * h  # o_proj
    )

    router_params = h * config.num_local_experts

    # SwiGLU expert: gate_up (w1+w3) + down (w2), per expert per layer.
    expert_ffn_params = 2 * h * config.intermediate_size + config.intermediate_size * h

    per_layer = attn_params + router_params + config.num_experts_per_tok * expert_ffn_params
    return per_layer * config.num_hidden_layers


def compute_flops_per_step(config: NVMixtralConfig, total_tokens: int, seq_len: int) -> float:
    """Training fwd+bwd FLOPs for one optimizer step (MoE-aware formula).

    flops = 6 * N_active * total_tokens + 12 * L * total_tokens * seq_len * hidden_size

    Embeddings and lm_head are excluded from N_active.
    """
    n_active = compute_n_active(config)
    attn_flops = 12 * config.num_hidden_layers * total_tokens * seq_len * config.hidden_size
    matmul_flops = 6 * n_active * total_tokens
    return matmul_flops + attn_flops


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments (parallelism layout, precision, batch/seq, warmup/measure steps)."""
    parser = argparse.ArgumentParser(description="Mixtral native TE benchmark")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--precision", choices=["bf16", "mxfp8"], default="bf16")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=30)
    parser.add_argument("--quantized-model-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the standalone throughput benchmark for one (dp, ep) layout and precision."""
    args = parse_args()
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    if args.dp_size * args.ep_size != dist_config.world_size:
        raise ValueError(
            f"dp_size ({args.dp_size}) * ep_size ({args.ep_size}) must equal world_size ({dist_config.world_size})"
        )

    fp8_enabled = args.precision == "mxfp8"
    if fp8_enabled:
        _assert_fused_mxfp8_supported()

    fp8_recipe = None
    if fp8_enabled:
        fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    config = NVMixtralConfig.from_pretrained(
        args.config_path,
        dtype=torch.bfloat16,
        expert_ffn_mode="fused_grouped_mlp",
        expert_parallel_size=args.ep_size,
    )
    qmi_kwargs = {"enabled": False}
    if fp8_enabled and args.quantized_model_init:
        # Benchmark only measures throughput (random weights, no resume), so we do NOT preserve the
        # high-precision init value: the training path clears it after seeding the fp32 master
        # (_init_master_weights_from_high_precision), but the benchmark never consumes it, so
        # preserving would leak a full extra fp32 copy of every param for the whole run.
        qmi_kwargs = {"enabled": True}

    with transformer_engine.pytorch.quantized_model_init(recipe=fp8_recipe, **qmi_kwargs):
        model = NVMixtralForCausalLM(config, fp8_recipe=fp8_recipe)

    model = model.to(device=device, dtype=torch.bfloat16)
    mesh = build_mesh_and_wrap(model, dp_size=args.dp_size, ep_size=args.ep_size)

    optimizer = FusedAdam(model.parameters(), master_weights=True, lr=1e-4, betas=(0.9, 0.95))

    vocab_size = config.vocab_size
    micro_batch = args.micro_batch_size
    seq_len = args.seq_len

    def make_batch() -> dict[str, torch.Tensor]:
        input_ids = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    # Warmup (untimed).
    for _ in range(args.warmup_steps):
        batch = make_batch()
        outputs = model(**batch)
        loss = outputs.loss
        assert loss is not None
        loss.backward()
        if args.ep_size > 1:
            all_reduce_dense_grads_over_ep(model, mesh["ep"].get_group())
        _clip_grad_norm_mixed(
            model,
            max_norm=1.0,
            ep_group=mesh["ep"].get_group(),
            dp_group=mesh["dp"].get_group(),
        )
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()

    step_times: list[float] = []
    for _ in range(args.measured_steps):
        batch = make_batch()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()

        outputs = model(**batch)
        loss = outputs.loss
        assert loss is not None
        loss.backward()
        if args.ep_size > 1:
            all_reduce_dense_grads_over_ep(model, mesh["ep"].get_group())
        _clip_grad_norm_mixed(
            model,
            max_norm=1.0,
            ep_group=mesh["ep"].get_group(),
            dp_group=mesh["dp"].get_group(),
        )
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        dist.barrier()
        step_times.append(time.perf_counter() - t0)

    mean_step_time = sum(step_times) / len(step_times)
    tokens_per_step = micro_batch * seq_len * args.dp_size
    tokens_per_sec_per_gpu = tokens_per_step / mean_step_time / dist_config.world_size
    flops_per_step = compute_flops_per_step(config, total_tokens=tokens_per_step, seq_len=seq_len)
    tflops_per_gpu = flops_per_step / dist_config.world_size / mean_step_time / 1e12
    peak = PEAK_FLOPS_FP8 if fp8_enabled else PEAK_FLOPS_BF16
    mfu_pct = 100.0 * (tflops_per_gpu * 1e12) / peak

    config_label = os.path.basename(args.config_path.rstrip("/"))
    precision_label = "mxfp8-autocast" if fp8_enabled and not args.quantized_model_init else args.precision
    if fp8_enabled and args.quantized_model_init:
        precision_label = "mxfp8-qmi"

    result_line = (
        f"{config_label} | {precision_label} | dp={args.dp_size} | ep={args.ep_size} | "
        f"{tokens_per_sec_per_gpu:.1f} tokens/s/gpu | {tflops_per_gpu:.2f} TFLOP/s/gpu | {mfu_pct:.1f}% MFU"
    )

    if dist_config.is_main_process():
        print(result_line)
        print(
            f"  step_time={mean_step_time:.3f}s  tokens/step={tokens_per_step}  "
            f"batch x seq={micro_batch}x{seq_len}  N_active={compute_n_active(config):,}  "
            f"peak={'FP8 4.5PFLOP/s' if fp8_enabled else 'BF16 2.25PFLOP/s'}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
