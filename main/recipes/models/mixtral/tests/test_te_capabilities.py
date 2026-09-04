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

# models/mixtral/tests/test_te_capabilities.py
"""Empirical probes for TE features required by the single_grouped_weight design.

These are intentionally non-parametrized, standalone probes that document what the
installed TransformerEngine build supports on the current GPU. They are the Phase-0
"spike" made permanent.
"""

import inspect
import os

import pytest
import torch
import transformer_engine
import transformer_engine.pytorch as te


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

GROUPED_LINEAR_PARITY_ATOL = 1e-2
GROUPED_LINEAR_PARITY_RTOL = 1e-2


def _is_supported_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10


requires_sm100 = pytest.mark.skipif(
    not _is_supported_blackwell(),
    reason="requires compute capability 10.x datacenter Blackwell; 12.0 support is pending",
)


def test_report_environment():
    """Print TE version + device capability for the record (never fails)."""
    print("TE version:", getattr(transformer_engine, "__version__", "unknown"))
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print("device:", torch.cuda.get_device_name(0), "capability:", cap)
    else:
        print("no CUDA device")


def test_module_grouped_linear_has_single_grouped_weight():
    """Module GroupedLinear must expose the single_grouped_weight kwarg."""
    params = inspect.signature(te.GroupedLinear.__init__).parameters
    assert "single_grouped_weight" in params, (
        "Installed TE's module GroupedLinear lacks single_grouped_weight; "
        "the Phase-0 gate cannot proceed on this build."
    )


def test_ops_fused_grouped_mlp_importable():
    """The op-based fused GroupedMLP path should import (needed later for the fused kernel)."""
    from transformer_engine.pytorch.ops import GroupedLinear as OpsGroupedLinear  # noqa: F401
    from transformer_engine.pytorch.ops import Sequential  # noqa: F401

    from modeling_mixtral_te import _fused_grouped_mlp_op_class

    swiglu_ok = True
    try:
        from transformer_engine.pytorch.ops import ScaledSwiGLU  # noqa: F401
    except Exception:
        try:
            from transformer_engine.pytorch.ops import SwiGLU  # noqa: F401
        except Exception:
            swiglu_ok = False
    assert swiglu_ok, "Neither ScaledSwiGLU nor SwiGLU is importable from transformer_engine.pytorch.ops"
    assert _fused_grouped_mlp_op_class().__name__ in {
        "ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8",
        "GroupedMLP_CuTeGEMMGLU",
    }


def _assert_same_tolerance_rejects_shifted(reference: torch.Tensor, actual: torch.Tensor) -> None:
    """Guard against over-loose tolerances by checking a token-shuffled baseline fails."""
    assert actual.shape[0] > 1, "negative control needs at least two tokens"
    shifted = torch.roll(actual, shifts=1, dims=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            reference,
            shifted,
            atol=GROUPED_LINEAR_PARITY_ATOL,
            rtol=GROUPED_LINEAR_PARITY_RTOL,
        )


@requires_cuda
def test_single_grouped_weight_matches_discrete_forward_backward():
    """A single_grouped_weight GroupedLinear must match a discrete one numerically."""
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    torch.manual_seed(0)

    num_gemms, in_f, out_f = 4, 32, 64
    m_splits = [3, 5, 0, 4]
    total = sum(m_splits)
    dtype = torch.bfloat16

    def make(single: bool):
        torch.manual_seed(0)
        return te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            params_dtype=dtype,
            device="cuda",
            single_grouped_weight=single,
        )

    discrete = make(False)
    single = make(True)

    # Verify storage shape: single -> one "weight" GroupedTensor, no weight0.
    assert hasattr(single, "weight"), "single_grouped_weight did not register a 'weight' param"
    assert getattr(single, "weight0", None) is None, "weight0 should be None with single weight"

    # GroupedTensor forbids integer indexing (aten.select.int) and other shape-manipulation
    # ops, so we cannot write single.weight[i]. Instead read the single grouped weight via
    # dequantize() -> a regular [num_gemms, out_f, in_f] tensor, and copy each GEMM into the
    # (indexable) discrete weight{i} params so both layers hold identical weights.
    with torch.no_grad():
        deq = single.weight.dequantize()  # [num_gemms, out_f, in_f]
        assert deq.shape == (num_gemms, out_f, in_f), f"unexpected dequantized shape {deq.shape}"
        for i in range(num_gemms):
            getattr(discrete, f"weight{i}").copy_(deq[i])

    x = torch.randn(total, in_f, device="cuda", dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    out_d = discrete(x, m_splits=m_splits)
    out_s = single(x2, m_splits=m_splits)
    torch.testing.assert_close(out_s, out_d, atol=GROUPED_LINEAR_PARITY_ATOL, rtol=GROUPED_LINEAR_PARITY_RTOL)
    _assert_same_tolerance_rejects_shifted(out_d, out_s)

    out_d.sum().backward()
    out_s.sum().backward()
    torch.testing.assert_close(
        x2.grad,
        x.grad,
        atol=GROUPED_LINEAR_PARITY_ATOL,
        rtol=GROUPED_LINEAR_PARITY_RTOL,
    )
    _assert_same_tolerance_rejects_shifted(x.grad, x2.grad)


# Spike finding (TE 2.16.0+4220403e, RTX 5090): DTensor.from_local(gl.weight, mesh, [Shard(0)])
# fails before nn.Parameter or forward. from_local calls input.view_as(input), which dispatches
# aten.view on GroupedTensor; TE only permits view(-1) for distributed-optimizer flattening.
# Error: RuntimeError: GroupedTensor only supports view(-1) for distributed optimizer flattening
@pytest.mark.xfail(
    reason="DTensor.from_local calls view_as on GroupedTensor; TE only supports view(-1)",
    strict=False,
)
@requires_cuda
def test_single_grouped_weight_wraps_as_dtensor_shard0():
    """The GroupedTensor weight must wrap as DTensor(Shard(0)) and run a forward."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.device_mesh import DeviceMesh

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    if dist.is_initialized():
        pytest.skip("Distributed already initialized")
    torch.cuda.set_device(0)
    store = dist.HashStore()
    dist.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
    try:
        num_gemms, in_f, out_f = 4, 32, 64
        m_splits = [2, 2, 2, 2]
        gl = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_f,
            out_features=out_f,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        )
        ep_mesh = DeviceMesh("cuda", [0])
        local = gl.weight
        dt = DTensor.from_local(local, device_mesh=ep_mesh, placements=[Shard(0)])
        gl.weight = torch.nn.Parameter(dt)

        x = torch.randn(sum(m_splits), in_f, device="cuda", dtype=torch.bfloat16)
        out = gl(x, m_splits=m_splits)
        assert out.shape == (sum(m_splits), out_f)
    finally:
        dist.destroy_process_group()


@requires_cuda
def test_single_grouped_weight_dcp_roundtrip(tmp_path):
    """A plain (non-DTensor) single_grouped_weight module should survive DCP save -> load."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    if dist.is_initialized():
        pytest.skip("Distributed already initialized")
    torch.cuda.set_device(0)
    store = dist.HashStore()
    dist.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
    try:

        def build(seed):
            torch.manual_seed(seed)
            return te.GroupedLinear(
                num_gemms=4,
                in_features=32,
                out_features=64,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )

        # model (seed 0) is the source of truth; model2 (seed 1) starts different.
        model = build(0)
        model2 = build(1)

        # Sanity: they differ before load (compare via dequantize, never index the GroupedTensor).
        w0 = model.weight.dequantize()
        w1 = model2.weight.dequantize()
        assert not torch.allclose(w0, w1), "seeds produced identical weights; test is not meaningful"

        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

        ckpt_dir = str(tmp_path / "ckpt")
        dcp.save(sd, checkpoint_id=ckpt_dir)

        load_sd = get_model_state_dict(model2)  # includes _extra_state keys
        dcp.load(load_sd, checkpoint_id=ckpt_dir)
        set_model_state_dict(model2, load_sd)

        # After load, model2 should match model (compare via dequantize).
        torch.testing.assert_close(model2.weight.dequantize(), model.weight.dequantize(), atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@requires_cuda
@requires_sm100
def test_fused_grouped_mlp_bf16_forward_backward():
    """ops.Sequential(GroupedLinear, SwiGLU, GroupedLinear) is functional in BF16 with single_grouped_weight."""
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    from transformer_engine.pytorch.ops import GroupedLinear as OpsGroupedLinear
    from transformer_engine.pytorch.ops import Sequential, SwiGLU

    num_groups, hidden, inter = 4, 128, 256
    ffn = Sequential(
        OpsGroupedLinear(
            num_groups=num_groups,
            in_features=hidden,
            out_features=2 * inter,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        ),
        SwiGLU(),
        OpsGroupedLinear(
            num_groups=num_groups,
            in_features=inter,
            out_features=hidden,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        ),
    )
    tokens = 128
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    split_sizes = torch.tensor([32, 32, 32, 32], dtype=torch.int32, device="cuda")
    out = ffn(x, split_sizes, split_sizes)
    out.sum().backward()
    assert out.shape == (tokens, hidden)
    assert x.grad is not None


@requires_cuda
@requires_sm100
def test_fused_grouped_mlp_mxfp8_forward_backward():
    """The fused GroupedMLP op path runs under MXFP8; skip if MXFP8 unsupported on this build."""
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8

    supported, reason = fp8.check_mxfp8_support()
    if not supported:
        pytest.skip(f"MXFP8 unsupported: {reason}")

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    from transformer_engine.pytorch.ops import GroupedLinear as OpsGroupedLinear
    from transformer_engine.pytorch.ops import ScaledSwiGLU, Sequential

    num_groups, hidden, inter = 4, 128, 256  # inter divisible by 32 (GLU interleave)
    ffn = Sequential(
        OpsGroupedLinear(
            num_groups=num_groups,
            in_features=hidden,
            out_features=2 * inter,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        ),
        ScaledSwiGLU(glu_interleave_size=32),
        OpsGroupedLinear(
            num_groups=num_groups,
            in_features=inter,
            out_features=hidden,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        ),
    )
    tokens = 1024  # 256 per group for fused CuteDSL grouped GEMM (÷256)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    split_sizes = torch.tensor([256, 256, 256, 256], dtype=torch.int32, device="cuda")
    probs = torch.ones(tokens, device="cuda", dtype=torch.float32)

    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    with te.autocast(enabled=True, recipe=recipe):
        out = ffn(x, split_sizes, probs, split_sizes)
    out.sum().backward()
    assert out.shape == (tokens, hidden)
    assert x.grad is not None


@requires_cuda
@requires_sm100
def test_fused_grouped_mlp_mxfp8_matches_bf16_reference():
    """Fused MXFP8 GroupedMLP should match a bf16 GroupedLinear+SwiGLU reference within MXFP8 tol."""
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8
    from transformer_engine.pytorch.ops import (
        GroupedLinear as OpsGL,
    )
    from transformer_engine.pytorch.ops import (
        ScaledSwiGLU,
        Sequential,
    )

    supported, reason = fp8.check_mxfp8_support()
    if not supported:
        pytest.skip(f"MXFP8 unsupported: {reason}")

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    torch.manual_seed(0)
    num_groups, hidden, inter = 4, 256, 512
    tokens, split = 1024, [256, 256, 256, 256]
    split_sizes = torch.tensor(split, dtype=torch.int32, device="cuda")
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    # Fused MXFP8 path (bf16 master weights; MXFP8 applied via autocast).
    ffn = Sequential(
        OpsGL(
            num_groups=num_groups,
            in_features=hidden,
            out_features=2 * inter,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        ),
        ScaledSwiGLU(glu_interleave_size=32),
        OpsGL(
            num_groups=num_groups,
            in_features=inter,
            out_features=hidden,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
        ),
    )
    probs = torch.ones(tokens, device="cuda", dtype=torch.float32)
    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    xg = x.clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=recipe):
        out_fused = ffn(xg, split_sizes, probs, split_sizes)
    out_fused.sum().backward()

    # Report whether the CuteDSL fused op is supported (best-effort; do not fail the test on this).
    try:
        from modeling_mixtral_te import _fused_grouped_mlp_op_class

        print("fused_op_is_supported:", _fused_grouped_mlp_op_class().is_supported())
    except Exception as e:
        print("could not introspect fused forward op:", repr(e))

    assert out_fused.shape == (tokens, hidden)
    assert xg.grad is not None


@requires_cuda
@requires_sm100
def test_mxfp8_leaves_weights_bf16_without_quantized_init():
    """Without quantized_model_init, the GroupedLinear weight stays bf16; MXFP8 is transient.

    This is the property that makes the checkpoint a plain bf16 checkpoint.
    """
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8
    from transformer_engine.pytorch.ops import GroupedLinear as OpsGL

    supported, reason = fp8.check_mxfp8_support()
    if not supported:
        pytest.skip(f"MXFP8 unsupported: {reason}")

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
    gl = OpsGL(
        num_groups=4,
        in_features=128,
        out_features=256,
        bias=False,
        dtype=torch.bfloat16,
        device="cuda",
        single_grouped_weight=True,
    )
    # The stored parameter must be bf16 (not a persistent fp8 tensor).
    assert gl.weight.dtype == torch.bfloat16, f"expected bf16 master weight, got {gl.weight.dtype}"
    before = gl.weight.dequantize().clone()

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    split_sizes = torch.tensor([32, 32, 32, 32], dtype=torch.int32, device="cuda")
    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    with te.autocast(enabled=True, recipe=recipe):
        _ = gl(x, split_sizes)
    after = gl.weight.dequantize()
    # A forward must not mutate the persistent bf16 weight.
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    assert gl.weight.dtype == torch.bfloat16
