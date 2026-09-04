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

"""2D (dp, ep) device mesh setup and collectives for selective FSDP2 wrapping."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


# Discrete expert weight module names; everything else is a "dense" (non-expert) param.
_EXPERT_MARKERS = (".experts_gate_up.", ".experts_down.")


def _is_expert_param(name: str) -> bool:
    """Return whether a named parameter is an expert weight (owned/partitioned over ``ep``).

    Experts are excluded from the dense (non-expert) gradient all-reduce over ``ep``: each rank owns
    disjoint experts and, via the token all-to-all, already receives every token routed to them, so
    their gradients are owner-complete. Only the *replicated* dense params need the ep all-reduce.
    """
    return any(marker in name for marker in _EXPERT_MARKERS)


def all_reduce_dense_grads_over_ep(model, ep_group) -> None:
    """Average non-expert (dense) gradients across the ``ep`` group.

    Dense params are replicated across ``ep`` (every ``ep`` rank holds the same dense params) but
    each ``ep`` rank processes different data, so their dense gradients differ and must be averaged
    over ``ep`` to keep the replicated dense params in sync. This supplies the ``ep`` half of the
    dense data-parallel reduction in both modes that have an ``ep`` dimension:

    - ``dp == 1`` (EP-only): dense is unwrapped/plain and gets *no* FSDP reduction, so this
      all-reduce over ``ep`` (``= world``) is the entire dense data-parallel reduction.
    - ``dp > 1`` (EP+FSDP2): dense is FSDP-reduced over ``dp`` already; this adds the ``ep`` half,
      giving a full reduction over ``dp x ep`` = world.

    Experts are EP-partitioned (each rank owns disjoint experts and, via the token all-to-all,
    already receives every token routed to them) so their gradients are owner-complete and skipped.
    No-op when ``ep_group`` is ``None`` or has size 1. Runs outside autograd (after grad
    accumulation, before clipping), so it introduces no cross-process-group ordering hazard.
    """
    if ep_group is None or ep_group.size() == 1:
        return
    ep_world = ep_group.size()
    for name, param in model.named_parameters():
        if param.grad is None or _is_expert_param(name):
            continue
        grad = param.grad
        local = grad.to_local() if isinstance(grad, DTensor) else grad
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=ep_group)
        local.div_(ep_world)


def clip_grad_norm_mixed(
    model: torch.nn.Module,
    max_norm: float,
    ep_group: dist.ProcessGroup | None = None,
    dp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Clip the global L2 norm of EP-local parameters mixed with FSDP2 DTensors.

    This is necessary whenever expert parallelism and FSDP2 coexist. Stock
    ``torch.nn.utils.clip_grad_norm_`` sees only local DTensor shards, while FSDP1's clipping API
    does not apply to composable FSDP2 or understand that expert and dense parameters have different
    ownership. We therefore complete the squared norms according to the recipe's 2D layout:

    - expert gradients own disjoint experts across ``ep`` and, for ``dp > 1``, disjoint FSDP shards
      across ``dp``; their squared norm is reduced over both groups;
    - dense gradients are FSDP-sharded across ``dp`` but already averaged and identical across
      ``ep`` by :func:`all_reduce_dense_grads_over_ep`, so their squared norm is reduced only over
      ``dp``.

    This helper can be removed if PyTorch exposes FSDP2/DTensor norm clipping with heterogeneous
    parameter placements. Until then the ownership policy is recipe-specific and must be explicit.
    """
    device = next(model.parameters()).device
    expert_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
    dense_sq_sum = torch.zeros((), device=device, dtype=torch.float32)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        local_norm = torch.linalg.vector_norm(grad, ord=2, dtype=torch.float32)
        sq_sum = local_norm.square()
        if _is_expert_param(name):
            expert_sq_sum += sq_sum
        else:
            dense_sq_sum += sq_sum

    if ep_group is not None and ep_group.size() > 1:
        dist.all_reduce(expert_sq_sum, op=dist.ReduceOp.SUM, group=ep_group)

    # Once experts have been completed over EP, both classes need exactly the same DP reduction.
    # Combining the scalar sums keeps clipping to one collective per non-trivial mesh dimension.
    total_sq_sum = expert_sq_sum + dense_sq_sum
    if dp_group is not None and dp_group.size() > 1:
        dist.all_reduce(total_sq_sum, op=dist.ReduceOp.SUM, group=dp_group)

    total_norm = torch.sqrt(total_sq_sum)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                if isinstance(param.grad, DTensor):
                    param.grad = param.grad * clip_coef
                else:
                    param.grad.mul_(clip_coef)
    return total_norm


def build_mesh_and_wrap(model, dp_size: int, ep_size: int) -> DeviceMesh:
    """Build a 2D (dp, ep) mesh, activate EP, and FSDP2-wrap for the chosen sharding mode.

    The rule is selected purely by the ``(dp_size, ep_size)`` factorization of world size. In every
    mode the **dense** (non-expert) params are data-parallel over the *full world* — dense params
    are replicated across ``ep`` and their gradients must be reduced over every rank — while the
    **experts** are expert-parallel over ``ep`` and, when ``dp > 1``, additionally FSDP2-sharded
    over ``dp``.

    - **EP-only ``(dp=1, ep>1)``** — no FSDP2 wrapping at all. Experts are EP-local plain
      ``weight{i}``; dense params are plain and replicated, kept in sync by an explicit all-reduce
      of dense gradients over ``ep`` each step (``all_reduce_dense_grads_over_ep``).
    - **FSDP2-only ``(dp>1, ep=1)``** — whole-layer wrap over ``dp`` (``= world``); everything is
      FSDP2-sharded and reduced over ``dp``.
    - **EP+FSDP2 ``(dp>1, ep>1)``** — whole-layer wrap over ``dp`` shards experts+dense over ``dp``;
      dense is only replicated across ``ep``, so its gradients are additionally all-reduced over
      ``ep`` each step. Full-world dense reduction = FSDP reduce over ``dp`` + all-reduce over ``ep``.

    Load-bearing details:

    - When ``dp > 1``, the **whole decoder layer** is wrapped (never the leaf ``GroupedLinear``): the
      layer's pre-forward all-gather materializes the experts before TE's fused ``GroupedMLP`` kernel
      reads them. Leaf-level ``fully_shard(experts_gate_up)`` crashes (``cast_gated_tma`` /
      illegal memory access) because the ops-fuser bypasses the leaf module.
    - When ``dp == 1``, we deliberately do **not** FSDP2-wrap anything. A size-1 ``dp`` mesh shards
      nothing yet crashes TE's MXFP8 all-gather. Wrapping dense over a *full-world* mesh instead
      (size ``ep``) would work numerically but introduces world-group FSDP collectives that
      interleave with the ep-group token all-to-all in backward; since ``world == ep`` at ``dp=1``
      (two process groups over the same ranks), routing-dependent backward graphs desync the two
      groups' collectives and hang. Keeping dense unwrapped and syncing it via a single explicit
      ep all-reduce (outside autograd) avoids the cross-group ordering hazard entirely.

    Args:
        model: ``NVMixtralForCausalLM`` instance (unwrapped).
        dp_size: Data-parallel mesh dimension size.
        ep_size: Expert-parallel mesh dimension size.

    Returns:
        The 2D ``DeviceMesh`` with dimensions ``("dp", "ep")``.
    """
    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))

    # Set EP groups BEFORE FSDP2 wrapping: this puts the expert parameters/views into their final
    # expert-parallel layout (and, in grouped_linear mode, converts them to ep DTensors) before
    # `fully_shard` captures the module's parameters and installs its pre-forward all-gather hooks.
    # Wrapping first would let FSDP2 flatten the pre-EP weights and the layouts would disagree.
    # Trivial (no-op) when ep_size == 1.
    model.model.set_ep_groups(mesh["ep"].get_group(), mesh["ep"])

    # Reduce the MoE load-balancing statistics over the full world (= dp x ep = world_size). Because
    # data is sharded over the whole world, every rank holds distinct tokens, so the switch aux loss
    # must aggregate f_i / P_i across all ranks to reflect the global token distribution (reducing
    # over only dp would miss the ep ranks' distinct tokens). This reduction is forward-only /
    # collective-free in backward (see the detach trick in modeling_mixtral_te), so unlike a
    # world-group FSDP wrap it introduces no cross-group backward-ordering hazard.
    model.model.set_load_balance_group(dist.group.WORLD)

    if dp_size > 1:
        # Whole-layer wrap over dp shards experts + dense over dp; the root wrap captures the
        # remaining top-level params (embed_tokens, norm, lm_head). Dense grads are additionally
        # reduced over ep in the training loop when ep_size > 1.
        dp_mesh = mesh["dp"]
        for layer in model.model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)
    # dp_size == 1: no FSDP2 wrapping. Dense stays plain (synced via the ep all-reduce below when
    # ep_size > 1); experts stay EP-local. ep_size == 1 as well means a single process.

    return mesh
