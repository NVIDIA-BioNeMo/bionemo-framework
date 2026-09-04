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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: models/mixtral/grouped_dcp.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Consolidated DCP checkpointing for expert-parallel MoE expert weights.

WHY this module exists (what stock DCP can't do for us): PyTorch's Distributed Checkpoint (DCP) can
save/load/reshard the tensors it finds in a model's state dict, but our experts aren't in a form it
can reshard. On each rank the experts live as TE **discrete per-expert ``weight{i}`` tensors** (often
MXFP8-quantized), i.e. this rank's ``ep`` shard of a conceptual global ``[num_experts, ...]`` tensor
that DCP never sees as a single DTensor. Saving those raw keys would bake in the current ``(dp, ep)``
factorization and could not be loaded into a different layout. This module constructs the *missing*
global expert tensor — stack the local experts and wrap as ``DTensor(Shard(0))`` on the ``ep`` mesh —
so the on-disk checkpoint is layout-independent and DCP can reshard it into any other ``(dp, ep)``.

The model-weight path supports both single ``GroupedTensor`` weights
(``single_grouped_weight=True``) and discrete ``weight{i}`` per-expert params
(``fused_grouped_mlp`` / ``NVTE_GROUPED_LINEAR_SINGLE_PARAM=0``). The optimizer-state path supports
only the discrete representation because TE 2.16 FusedAdam fails when updating a GroupedTensor.

We only need a global, reshardable representation at save/load time. For GroupedTensor /
MXFP8Tensor weights we operate on the *dequantized* bf16 view: at save, stack local expert shards
and wrap as ``DTensor(Shard(0))`` on the EP mesh; at load, reshard back and ``copy_`` into the
live weights (re-quantizing MXFP8 params on copy). FusedAdam optimizer state
(``master_param``, ``exp_avg``, ``exp_avg_sq``) is consolidated the same way for discrete experts.

TE's experimental ``single_grouped_weight`` does not by itself make this module removable. It
combines each rank's local experts into a TE ``GroupedTensor``, not an ordinary global tensor or
DTensor. In TE 2.16, ``DTensor.from_local`` calls ``view_as`` and GroupedTensor rejects that shape
operation; FSDP2 therefore cannot shard it over ``dp``. Pure EP model-weight DCP does work after
dequantizing it to an ordinary tensor, but FusedAdam fails on the live GroupedTensor and
``quantized_model_init`` does not persistently MXFP8-quantize it.

This module can be retired if TE makes the grouped parameter DTensor/FSDP2/FusedAdam-compatible (and
persistently quantizable), exposes an ordinary expert-sharded tensor instead, or PyTorch DCP gains a
supported adapter for logical tensors assembled from tensor-subclass parameters. Today neither
package has enough information to infer the discrete ``weight{i}``-to-global-expert mapping.

Expert weights come in two representations, handled transparently by ``_full_local_expert`` /
``_copy_into_expert``:

- **EP-local plain (dp==1):** each ``weight{i}`` is a plain (possibly MXFP8) tensor.
- **FSDP2-sharded (dp>1):** each ``weight{i}`` is a ``DTensor(Shard(0))`` on the dp mesh (a partial
  expert). Consolidation first all-gathers over dp (``full_tensor()``) to reconstruct the whole
  locally-owned expert, then places the stacked experts as ``DTensor(Shard(0))`` on the EP mesh;
  load reverses (reshard over ep, then re-scatter into each dp shard).
"""

from __future__ import annotations

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.device_mesh import DeviceMesh


# Expert weight module names on each NVMixtralSparseMoeBlock (discrete weight{i} or single GroupedTensor weight).
_EXPERT_WEIGHT_ATTRS = ("experts_gate_up", "experts_down")
_OPTIMIZER_STATE_ATTRS = ("master_param", "exp_avg", "exp_avg_sq")


def _is_discrete_grouped_linear(gl: torch.nn.Module) -> bool:
    """True when the module exposes per-expert ``weight{i}`` params instead of one GroupedTensor."""
    return gl._parameters.get("weight0") is not None


def _count_discrete_expert_weights(gl: torch.nn.Module) -> int:
    """Count ``weight{i}`` attributes on a discrete GroupedLinear."""
    count = 0
    while hasattr(gl, f"weight{count}"):
        count += 1
    return count


def _discrete_expert_weights(gl: torch.nn.Module, num_local: int) -> list[torch.Tensor]:
    return [getattr(gl, f"weight{i}") for i in range(num_local)]


def _full_local_expert(w: torch.Tensor) -> torch.Tensor:
    """Return a plain, dp-gathered bf16 view of one expert weight/optimizer-state tensor.

    Handles both expert representations transparently:

    - **FSDP2-sharded (dp>1):** ``w`` is a ``DTensor(Shard(0))`` on the dp mesh (a partial expert).
      ``full_tensor()`` all-gathers over dp to reconstruct the whole locally-owned expert; TE's
      FSDP hooks reconstruct the quantized tensor, which we then dequantize.
    - **EP-local plain (dp==1, or non-FSDP callers):** dequantize / detach directly.
    """
    if isinstance(w, DTensor):
        w = w.full_tensor()
    return w.dequantize() if hasattr(w, "dequantize") else w.detach()


def _copy_into_expert(dst: torch.Tensor, full_expert: torch.Tensor) -> None:
    """Write a full (global) expert tensor into a possibly dp-sharded destination.

    - **FSDP2-sharded (dp>1):** ``dst`` is a ``DTensor(Shard(0))`` on the dp mesh; re-shard
      ``full_expert`` to the same dp layout and ``copy_`` (re-quantizes MXFP8 params on copy).
    - **EP-local plain (dp==1, or non-FSDP callers):** copy the whole expert directly.
    """
    if isinstance(dst, DTensor):
        src = distribute_tensor(full_expert.to(dst.dtype), dst.device_mesh, dst.placements)
        dst.copy_(src)
    else:
        dst.copy_(full_expert)


def _stack_discrete_expert_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-expert tensors into ``[num_local, out, in]`` (dp-gathered when FSDP2-sharded)."""
    return torch.stack([_full_local_expert(w) for w in tensors], dim=0).contiguous()


def build_ep_sharded_state_dict(model: torch.nn.Module, ep_mesh: DeviceMesh) -> dict[str, torch.Tensor | DTensor]:
    """Build the consolidated, EP-reshardable state dict used for checkpoint save/load.

    Each expert weight becomes a ``DTensor(Shard(0))`` of its bf16-dequantized local shard on the ep
    mesh; every other parameter is included as-is (already a plain tensor or FSDP2 DTensor).
    """
    sd: dict[str, torch.Tensor | DTensor] = {}
    expert_prefixes: set[str] = set()
    for name, module in model.named_modules():
        for attr in _EXPERT_WEIGHT_ATTRS:
            gl = getattr(module, attr, None)
            if gl is None:
                continue
            prefix = f"{name}.{attr}"
            if _is_discrete_grouped_linear(gl):
                num_local = _count_discrete_expert_weights(gl)
                local_stack = _stack_discrete_expert_tensors(_discrete_expert_weights(gl, num_local))
                expert_prefixes.add(prefix)
                sd[f"{prefix}.weight"] = DTensor.from_local(local_stack, device_mesh=ep_mesh, placements=[Shard(0)])
            elif hasattr(gl, "weight"):
                expert_prefixes.add(prefix)
                local_bf16 = _full_local_expert(gl.weight)
                sd[f"{prefix}.weight"] = DTensor.from_local(
                    local_bf16.contiguous(), device_mesh=ep_mesh, placements=[Shard(0)]
                )
    # Non-expert params (embeddings, attention, norms, lm_head) — plain, replicated across EP.
    expert_keys = set(sd)
    for name, p in model.named_parameters():
        if name in expert_keys or name.endswith("_extra_state"):
            continue
        if any(name.startswith(pfx) for pfx in expert_prefixes):
            continue
        sd[name] = p.detach()
    return sd


def save_consolidated(model: torch.nn.Module, ep_mesh: DeviceMesh, ckpt_dir: str) -> None:
    """Save one consolidated, EP-reshardable checkpoint (bf16 expert weights)."""
    sd = build_ep_sharded_state_dict(model, ep_mesh)
    dcp.save(sd, checkpoint_id=ckpt_dir)


def load_consolidated(model: torch.nn.Module, ep_mesh: DeviceMesh, ckpt_dir: str) -> None:
    """Load a consolidated checkpoint into ``model``, resharding experts to this EP degree.

    Writes the resharded bf16 values back into the live (possibly MXFP8) expert weights.
    The state dict built from the *current* model is used as a set of DCP load *templates*: each
    expert entry is a ``DTensor(Shard(0))`` on this run's ``ep`` mesh, and its placement tells
    ``dcp.load`` how to reshard the global on-disk ``[num_experts, ...]`` tensor for this EP degree
    (this is how a checkpoint saved at one ``(dp, ep)`` loads into another). After load we ``copy_``
    the resharded bf16 values back into the live (possibly MXFP8) expert weights, re-quantizing.
    """
    sd = build_ep_sharded_state_dict(model, ep_mesh)  # DTensors act as load templates + placement
    dcp.load(sd, checkpoint_id=ckpt_dir)
    with torch.no_grad():
        for name, module in model.named_modules():
            for attr in _EXPERT_WEIGHT_ATTRS:
                gl = getattr(module, attr, None)
                if gl is None:
                    continue
                prefix = f"{name}.{attr}"
                key = f"{prefix}.weight"
                if _is_discrete_grouped_linear(gl):
                    if key not in sd:
                        continue
                    local = sd[key].to_local()
                    num_local = _count_discrete_expert_weights(gl)
                    for i in range(num_local):
                        _copy_into_expert(getattr(gl, f"weight{i}"), local[i])
                elif hasattr(gl, "weight"):
                    if key not in sd:
                        continue
                    local = sd[key].to_local()
                    _copy_into_expert(gl.weight, local)


def build_optimizer_ep_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ep_mesh: DeviceMesh,
) -> dict[str, DTensor]:
    """Return optimizer state for discrete expert weights as DTensor(Shard(0)) stacks."""
    sd: dict[str, DTensor] = {}
    for name, module in model.named_modules():
        for attr in _EXPERT_WEIGHT_ATTRS:
            gl = getattr(module, attr, None)
            if gl is None or not _is_discrete_grouped_linear(gl):
                continue
            num_local = _count_discrete_expert_weights(gl)
            prefix = f"{name}.{attr}"
            for state_name in _OPTIMIZER_STATE_ATTRS:
                locals_ = [
                    _full_local_expert(optimizer.state[getattr(gl, f"weight{i}")][state_name])
                    for i in range(num_local)
                ]
                stack = torch.stack(locals_, dim=0).contiguous()
                sd[f"{prefix}.{state_name}"] = DTensor.from_local(stack, device_mesh=ep_mesh, placements=[Shard(0)])
    return sd


def save_optimizer_consolidated(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ep_mesh: DeviceMesh,
    ckpt_dir: str,
) -> None:
    """Save consolidated, EP-reshardable FusedAdam state for discrete expert weights."""
    sd = build_optimizer_ep_state_dict(model, optimizer, ep_mesh)
    dcp.save(sd, checkpoint_id=ckpt_dir)


def load_optimizer_consolidated(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ep_mesh: DeviceMesh,
    ckpt_dir: str,
) -> None:
    """Load consolidated optimizer state, resharding to this EP degree."""
    sd = build_optimizer_ep_state_dict(model, optimizer, ep_mesh)
    dcp.load(sd, checkpoint_id=ckpt_dir)
    with torch.no_grad():
        for name, module in model.named_modules():
            for attr in _EXPERT_WEIGHT_ATTRS:
                gl = getattr(module, attr, None)
                if gl is None or not _is_discrete_grouped_linear(gl):
                    continue
                num_local = _count_discrete_expert_weights(gl)
                prefix = f"{name}.{attr}"
                for state_name in _OPTIMIZER_STATE_ATTRS:
                    stack = sd[f"{prefix}.{state_name}"].to_local()
                    for i in range(num_local):
                        _copy_into_expert(optimizer.state[getattr(gl, f"weight{i}")][state_name], stack[i])
