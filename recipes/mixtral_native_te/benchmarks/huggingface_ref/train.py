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

"""High-performance upstream Hugging Face Mixtral training benchmark.

This is deliberately independent of Transformer Engine. It uses the upstream
``transformers.MixtralForCausalLM`` implementation, Hugging Face native expert parallelism or
composable FSDP2, and full-graph TorchInductor compilation of every decoder block. BF16 uses
Hugging Face grouped GEMMs; MXFP8 uses TorchAO's differentiable grouped training primitive.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralExperts


def parse_args() -> argparse.Namespace:
    """Parse and validate benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data-file", default=None, help="Local parquet path or glob with a text column.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="HF native expert/tensor-parallel degree: either 1 (FSDP) or WORLD_SIZE.",
    )
    parser.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"),
        default="sdpa",
    )
    parser.add_argument(
        "--experts",
        choices=("grouped_mm", "batched_mm", "deepgemm", "sonicmoe"),
        default="grouped_mm",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-iterations", type=int, default=2000)
    parser.add_argument("--decay-iterations", type=int, default=58000)
    parser.add_argument("--min-learning-rate", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument(
        "--random-init", action="store_true", help="Construct from config instead of pretrained weights."
    )
    parser.add_argument("--config", default=None, help="Config path/model id used with --random-init.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable only if the larger gathered-parameter residency is faster and fits.",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Set to 0 to omit clipping (the TE reference clips at 1.0).",
    )
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.benchmark_steps < 1:
        parser.error("warmup-steps and benchmark-steps must both be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.ep_size not in (1, world_size):
        parser.error(
            f"ep-size must be 1 or WORLD_SIZE={world_size}; hybrid EP+FSDP requires a custom "
            "multidimensional mesh and is intentionally outside this reference"
        )
    if args.warmup_iterations < 0 or args.decay_iterations <= args.warmup_iterations:
        parser.error("decay-iterations must be greater than non-negative warmup-iterations")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error("min-learning-rate must be between zero and learning-rate")
    if not args.synthetic_data and not args.data_file:
        parser.error("--data-file is required unless --synthetic-data is set")
    if args.experts == "grouped_mm" and args.compile_mode not in ("default", "max-autotune-no-cudagraphs"):
        parser.error("grouped_mm does not support CUDA graphs")
    return args


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize the rank-local CUDA device and NCCL process group."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def _apply_mxfp8(model: MixtralForCausalLM) -> None:
    """Use TorchAO's differentiable MXFP8 expert GEMMs with BF16 weight gradients."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS.register("mxfp8_grouped_mm", _mxfp8_grouped_mm_experts_forward)
    model.set_experts_implementation("mxfp8_grouped_mm")


def _mxfp8_grouped_mm_experts_forward(
    experts: MixtralExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run HF routing around TorchAO's Blackwell MXFP8 grouped training GEMM."""
    from torchao.prototype.moe_training import _to_mxfp8_then_scaled_grouped_mm

    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.flatten()
    expert_ids, permutation = torch.sort(top_k_index.flatten())
    selected_states = hidden_states[permutation // num_top_k]
    selected_sample_weights = sample_weights[permutation]
    sentinel_mask = expert_ids >= experts.num_experts

    tokens_per_expert = torch.histc(
        expert_ids.int(),
        bins=experts.num_experts,
        min=0,
        max=experts.num_experts - 1,
    )
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    padded_states, padded_indices, padded_offsets = _pad_expert_groups(
        selected_states,
        offsets,
        sentinel_mask,
    )
    padded_row_ids = torch.arange(padded_states.shape[0], device=padded_states.device)
    padded_sentinel_mask = padded_row_ids >= padded_offsets[-1]
    # The MX grouped GEMM leaves dInput outside its offsets undefined. This pre-mask is also a
    # backward barrier, preventing those skipped sentinel gradients from reaching hidden states.
    padded_states = padded_states.masked_fill(padded_sentinel_mask.unsqueeze(-1), 0.0)

    gate_up_weight = experts.gate_up_proj if experts.is_transposed else experts.gate_up_proj.transpose(-2, -1)
    projected = _to_mxfp8_then_scaled_grouped_mm(
        padded_states,
        gate_up_weight,
        offs=padded_offsets,
        wgrad_with_hp=True,
    )
    projected = projected.masked_fill(padded_sentinel_mask.unsqueeze(-1), 0.0)
    projected = experts._apply_gate(projected)
    down_weight = experts.down_proj if experts.is_transposed else experts.down_proj.transpose(-2, -1)
    projected = _to_mxfp8_then_scaled_grouped_mm(
        projected,
        down_weight,
        offs=padded_offsets,
        wgrad_with_hp=True,
    )
    projected = projected.masked_fill(padded_sentinel_mask.unsqueeze(-1), 0.0)
    projected = projected[padded_indices] * selected_sample_weights.unsqueeze(-1)
    projected = projected.masked_fill(sentinel_mask.unsqueeze(-1), 0.0)

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    projected = projected[inverse_permutation]
    return projected.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def _pad_expert_groups(
    values: torch.Tensor,
    offsets: torch.Tensor,
    sentinel_mask: torch.Tensor,
    alignment: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad real expert groups to the alignment required by the installed CuTeDSL kernels."""
    num_rows = values.shape[0]
    num_groups = offsets.shape[0]
    zero = offsets.new_zeros(1)
    starts = torch.cat((zero, offsets[:-1]))
    sizes = offsets - starts
    padded_sizes = torch.div(sizes + alignment - 1, alignment, rounding_mode="floor") * alignment
    padded_offsets = torch.cumsum(padded_sizes, dim=0, dtype=torch.int32)
    padded_starts = torch.cat((zero, padded_offsets[:-1]))

    rows = torch.arange(num_rows, device=values.device, dtype=offsets.dtype)
    group_ids = torch.searchsorted(offsets, rows, right=True).clamp(max=num_groups - 1)
    real_indices = rows - starts[group_ids] + padded_starts[group_ids]
    # EP sentinel rows are ignored by the grouped GEMM offsets. Place them in a unique static tail
    # and mask their uninitialized outputs before restoring token order.
    sentinel_indices = rows + num_groups * alignment
    padded_indices = torch.where(sentinel_mask, sentinel_indices, real_indices).to(torch.long)
    padded = values.new_zeros((num_rows + num_groups * alignment, values.shape[1]))
    padded = padded.index_copy(0, padded_indices, values)
    return padded, padded_indices, padded_offsets


def build_model(args: argparse.Namespace, device: torch.device) -> MixtralForCausalLM:
    """Load, shard, and compile upstream Mixtral."""
    common: dict[str, Any] = {
        "attn_implementation": args.attention,
        "experts_implementation": args.experts,
        "dtype": torch.bfloat16,
    }
    if args.random_init and args.ep_size > 1:
        raise ValueError("--random-init is not supported with HF native expert parallelism")
    if args.random_init:
        config = MixtralConfig.from_pretrained(args.config or args.model, **common)
        config.use_cache = False
        model = MixtralForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    elif args.ep_size > 1:
        from transformers.distributed.configuration_utils import DistributedConfig

        distributed_config = DistributedConfig(enable_expert_parallel=True)
        model = MixtralForCausalLM.from_pretrained(
            args.model,
            **common,
            distributed_config=distributed_config,
            local_files_only=True,
        )
        model.config.use_cache = False
    else:
        # Loading directly onto the rank-local GPU avoids materializing a ~90 GB CPU model per
        # process. Each GPU temporarily holds the full BF16 model, which fits on a 192 GB B200;
        # FSDP shards it immediately below.
        model = MixtralForCausalLM.from_pretrained(
            args.model,
            **common,
            device_map=device,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.config.use_cache = False

    if args.precision == "mxfp8":
        if torch.cuda.get_device_capability(device) < (10, 0):
            raise RuntimeError("TorchAO MXFP8 training requires an SM100+ GPU")
        if args.experts != "grouped_mm":
            raise ValueError("MXFP8 requires --experts grouped_mm")
        _apply_mxfp8(model)

    compile_mode = None if args.compile_mode == "default" else args.compile_mode
    if args.ep_size == 1:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.model.layers:
            fully_shard(
                layer,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=args.reshard_after_forward,
            )
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=args.reshard_after_forward,
        )
    for layer in model.model.layers:
        # FSDP pre/post-forward hooks intentionally live outside Dynamo. Compiling each layer's
        # forward after sharding lets those hooks gather/reshard around an enforced full-graph
        # decoder block. A single root graph would have to cross nested FSDP hooks and is rejected.
        layer.forward = torch.compile(
            layer.forward,
            backend="inductor",
            mode=compile_mode,
            fullgraph=True,
            dynamic=False,
        )
    model.train()
    return model


def _dataset_tokens(args: argparse.Namespace, data_rank: int, data_parallel_size: int) -> list[int]:
    import datasets

    paths = sorted(glob.glob(args.data_file))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched --data-file={args.data_file!r}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=True,
    )
    stream = datasets.load_dataset("parquet", data_files=paths, split="train", streaming=True)
    stream = stream.shard(num_shards=data_parallel_size, index=data_rank)
    needed = args.micro_batch_size * args.seq_len
    tokens: list[int] = []
    for row in stream:
        text = row.get(args.text_column)
        if text:
            tokens.extend(tokenizer(text, add_special_tokens=True)["input_ids"])
        if len(tokens) >= needed:
            return tokens[:needed]
    raise RuntimeError(f"Dataset shard {data_rank} yielded only {len(tokens)} tokens; need {needed}")


def make_batch(
    args: argparse.Namespace,
    config: MixtralConfig,
    data_rank: int,
    data_parallel_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create one fixed rank-local training batch."""
    if args.synthetic_data:
        generator = torch.Generator(device=device).manual_seed(args.seed + data_rank)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (args.micro_batch_size, args.seq_len),
            generator=generator,
            device=device,
        )
    else:
        input_ids = torch.tensor(_dataset_tokens(args, data_rank, data_parallel_size), dtype=torch.long).reshape(
            args.micro_batch_size, args.seq_len
        )
        input_ids = input_ids.pin_memory().to(device=device, non_blocking=True)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def distributed_max(value: float, device: torch.device) -> float:
    """Return the maximum scalar value across data-parallel ranks."""
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def learning_rate_scale(step: int, args: argparse.Namespace) -> float:
    """Linear warmup followed by cosine decay, matching the native TE benchmark."""
    if step < args.warmup_iterations:
        return (step + 1) / max(1, args.warmup_iterations)
    progress = min(
        1.0,
        (step - args.warmup_iterations) / (args.decay_iterations - args.warmup_iterations),
    )
    min_scale = args.min_learning_rate / args.learning_rate
    return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    """Run the distributed training benchmark."""
    args = parse_args()
    rank, world_size, device = init_distributed()
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True

    model = build_model(args, device)
    data_parallel_size = world_size if args.ep_size == 1 else 1
    data_rank = rank if args.ep_size == 1 else 0
    batch = make_batch(args, model.config, data_rank, data_parallel_size, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_scale(step, args),
    )
    torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print(
            "configuration "
            f"world_size={world_size} ep_size={args.ep_size} "
            f"data_parallel_size={data_parallel_size} model={args.model} seq_len={args.seq_len} "
            f"micro_batch_size={args.micro_batch_size} precision={args.precision} "
            f"attention={args.attention} experts={args.experts} compile_mode={args.compile_mode} "
            f"fullgraph=True reshard_after_forward={args.reshard_after_forward}",
            flush=True,
        )

    timings: list[float] = []
    last_loss = float("nan")
    total_steps = args.warmup_steps + args.benchmark_steps
    for step in range(total_steps):
        dist.barrier()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if loss is None:
            raise RuntimeError("Mixtral returned no loss")
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.clip_grad_norm,
                foreach=False,
            )
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        elapsed = distributed_max(time.perf_counter() - start, device)
        last_loss = loss.detach().float().item()
        if step >= args.warmup_steps:
            timings.append(elapsed)
        if rank == 0:
            phase = "warmup" if step < args.warmup_steps else "benchmark"
            print(
                f"{phase} step={step + 1}/{total_steps} loss={last_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} step_time_s={elapsed:.6f}",
                flush=True,
            )

    median_time = statistics.median(timings)
    mean_time = statistics.mean(timings)
    tokens_per_data_parallel_rank = args.micro_batch_size * args.seq_len
    global_tokens = tokens_per_data_parallel_rank * data_parallel_size
    peak_gb = distributed_max(torch.cuda.max_memory_allocated(device) / 1024**3, device)
    result = {
        "world_size": world_size,
        "ep_size": args.ep_size,
        "data_parallel_size": data_parallel_size,
        "model": args.model,
        "seq_len": args.seq_len,
        "micro_batch_size": args.micro_batch_size,
        "precision": args.precision,
        "attention": args.attention,
        "experts": args.experts,
        "compile_mode": args.compile_mode,
        "fullgraph_decoder_blocks": True,
        "reshard_after_forward": args.reshard_after_forward,
        "median_step_time_s": median_time,
        "mean_step_time_s": mean_time,
        "tokens_per_second_per_gpu": global_tokens / world_size / median_time,
        "global_tokens_per_second": global_tokens / median_time,
        "peak_memory_allocated_gb": peak_gb,
        "last_loss": last_loss,
    }
    if rank == 0:
        print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
