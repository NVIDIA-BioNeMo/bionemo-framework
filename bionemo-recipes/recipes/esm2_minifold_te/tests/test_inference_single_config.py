import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch._dynamo as dynamo


RECIPE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RECIPE_DIR))

PLAIN_SPEC = importlib.util.spec_from_file_location("plain_minifold_infer", RECIPE_DIR / "plain_minifold_infer.py")
PLAIN_MODULE = importlib.util.module_from_spec(PLAIN_SPEC)
assert PLAIN_SPEC is not None and PLAIN_SPEC.loader is not None
sys.modules[PLAIN_SPEC.name] = PLAIN_MODULE
PLAIN_SPEC.loader.exec_module(PLAIN_MODULE)

SCRIPT_SPEC = importlib.util.spec_from_file_location("inference_single_config", RECIPE_DIR / "inference_single_config.py")
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

import minifold_native_ext.ops as MNE_OPS


def _assert_no_graph_breaks(fn, *args) -> None:
    dynamo.reset()
    explain = dynamo.explain(fn)(*args)
    assert explain.graph_break_count == 0
    assert explain.graph_count == 1


def _clear_linear_block32_retry_cache() -> None:
    with MNE_OPS._LINEAR_BLOCK32_CACHE_LOCK:
        MNE_OPS._LINEAR_BLOCK32_CAPABILITY_CACHE.clear()


def test_fp8_roundtrip_shapes():
    x = torch.randn(2, 4, 4, 8, dtype=torch.bfloat16)
    x_fp8, scale = PLAIN_MODULE.quantize_to_fp8(x)
    x_deq = PLAIN_MODULE.dequantize_from_fp8(x_fp8, scale)
    assert x_fp8.shape == x.shape
    assert scale.shape == (2, 4, 4, 1)
    assert x_deq.shape == x.shape


def test_default_config_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args([])
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.seq_len == 256
    assert cfg.mbs == 16
    assert cfg.tri_impl == "bmm"
    assert cfg.pair_precision == "bf16"
    assert cfg.linear_precision == "bf16"
    assert cfg.use_torch_compile is False
    assert cfg.torch_compile_mode == "default"
    assert cfg.diagnostics_output is None


def test_compat_flag_maps_to_fp8_storage():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(["--fp8_activations"])
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.pair_precision == "fp8_storage"
    assert cfg.fp8_activations is True
    assert cfg.linear_precision == "bf16"


def test_linear_precision_flag_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(["--linear_precision", "fp8"])
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.linear_precision == "fp8"


def test_torch_compile_flags_load():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        [
            "--use_torch_compile",
            "--torch_compile_mode",
            "max-autotune",
            "--diagnostics_output",
            "/tmp/diag.json",
        ]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.use_torch_compile is True
    assert cfg.torch_compile_mode == "max-autotune"
    assert str(cfg.diagnostics_output) == "/tmp/diag.json"


def test_fp8_extreme_config_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        ["--pair_precision", "fp8_extreme", "--linear_precision", "fp8", "--tri_impl", "fp8_cublaslt"]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.pair_precision == "fp8_extreme"
    assert cfg.linear_precision == "fp8"
    assert cfg.tri_impl == "fp8_cublaslt"


def test_fp8_native_config_loads():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        ["--pair_precision", "fp8_native", "--linear_precision", "fp8", "--tri_impl", "fp8_cublaslt"]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    assert cfg.pair_precision == "fp8_native"
    assert cfg.linear_precision == "fp8"
    assert cfg.tri_impl == "fp8_cublaslt"


def test_fp8_extreme_validation_rejects_non_fp8_tri_backend():
    with pytest.raises(ValueError, match="pair_precision=fp8_extreme.*requires tri_impl"):
        PLAIN_MODULE.validate_fp8_extreme_configuration("fp8_extreme", "fp8", "bmm")


def test_fp8_native_validation_rejects_non_fp8_tri_backend():
    with pytest.raises(ValueError, match="pair_precision=fp8_native.*requires tri_impl"):
        PLAIN_MODULE.validate_fp8_extreme_configuration("fp8_native", "fp8", "bmm")


def test_render_markdown():
    args = SCRIPT_MODULE.build_arg_parser().parse_args(
        ["--seq_len", "1024", "--mbs", "8", "--tri_impl", "cublas_xbdnn", "--fp8_activations", "--linear_precision", "fp8"]
    )
    cfg = SCRIPT_MODULE.load_config(args)
    table = SCRIPT_MODULE.render_markdown(
        cfg,
        {
            "median_ms": 12.5,
            "proteins_per_sec": 640.0,
            "unpadded_tokens_per_sec": 655360.0,
            "peak_memory_allocated_gib": 3.25,
        },
    )
    assert "| cublas_xbdnn | fp8_storage | fp8 | 1024 | 8 | 12.50 |" in table
    assert "fp8_storage" in table
    assert "linear=fp8" in table


def test_quantized_pair_tensor_storage_accounting():
    x = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16)
    q = PLAIN_MODULE.QuantizedPairTensor.from_tensor(x)
    assert q.quantized_bytes == q.payload.numel() * q.payload.element_size()
    assert q.scale_bytes == q.scale.numel() * q.scale.element_size()


def test_linear_block32_backend_retries_and_caches(monkeypatch):
    class StubExtension:
        def __init__(self):
            self.calls: list[tuple[bool, bool]] = []

        def linear_block32_fused(
            self,
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        ):
            self.calls.append((bool(direct_fp8_output), bool(fuse_bias_epilogue)))
            if fuse_bias_epilogue:
                raise RuntimeError("cuBLASLt error at cublasLtMatmulAlgoGetHeuristic(test)")
            return MNE_OPS._linear_payload_and_scale(a, int(b_t.shape[1]))

    stub = StubExtension()
    monkeypatch.setattr(MNE_OPS, "_C", stub)
    _clear_linear_block32_retry_cache()

    a = torch.randn(1, 8, 160, dtype=torch.float32).to(torch.float8_e4m3fn)
    b_t = torch.randn(1, 640, 160, dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.randn(1, 8, 5, dtype=torch.float32)
    b_scale = torch.randn(1, 640, 5, dtype=torch.float32)
    bias = torch.randn(640, dtype=torch.bfloat16)

    payload, scale = MNE_OPS._call_linear_block32_backend(
        a,
        b_t,
        a_scale,
        b_scale,
        bias,
        "bfloat16",
        True,
        False,
        True,
        None,
        None,
    )
    assert payload.shape == (1, 8, 640)
    assert scale.shape == (1, 8, 20)
    assert stub.calls == [(False, True), (False, False)]

    payload, scale = MNE_OPS._call_linear_block32_backend(
        a,
        b_t,
        a_scale,
        b_scale,
        bias,
        "bfloat16",
        True,
        False,
        True,
        None,
        None,
    )
    assert payload.shape == (1, 8, 640)
    assert scale.shape == (1, 8, 20)
    assert stub.calls == [(False, True), (False, False), (False, False)]


def test_linear_block32_backend_proactively_demotes_transition_fc1_epilogue(monkeypatch):
    class StubExtension:
        def __init__(self):
            self.calls: list[tuple[bool, bool]] = []

        def linear_block32_fused(
            self,
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        ):
            self.calls.append((bool(direct_fp8_output), bool(fuse_bias_epilogue)))
            return MNE_OPS._linear_payload_and_scale(a, int(b_t.shape[1]))

    stub = StubExtension()
    monkeypatch.setattr(MNE_OPS, "_C", stub)
    _clear_linear_block32_retry_cache()

    a = torch.randn(1, 8, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    b_t = torch.randn(1, 512, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.randn(1, 8, 4, dtype=torch.float32)
    b_scale = torch.randn(1, 512, 4, dtype=torch.float32)
    bias = torch.randn(512, dtype=torch.bfloat16)

    payload, scale = MNE_OPS._call_linear_block32_backend(
        a,
        b_t,
        a_scale,
        b_scale,
        bias,
        "bfloat16",
        True,
        False,
        True,
        None,
        None,
    )

    assert payload.shape == (1, 8, 512)
    assert scale.shape == (1, 8, 16)
    assert stub.calls == [(False, False)]


def test_linear_block32_backend_proactively_demotes_transition_fc2_direct_output(monkeypatch):
    class StubExtension:
        def __init__(self):
            self.calls: list[tuple[bool, bool]] = []

        def linear_block32_fused(
            self,
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        ):
            self.calls.append((bool(direct_fp8_output), bool(fuse_bias_epilogue)))
            return MNE_OPS._linear_payload_and_scale(a, int(b_t.shape[1]))

    stub = StubExtension()
    monkeypatch.setattr(MNE_OPS, "_C", stub)
    _clear_linear_block32_retry_cache()

    a = torch.randn(1, 8, 512, dtype=torch.float32).to(torch.float8_e4m3fn)
    b_t = torch.randn(1, 128, 512, dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.randn(1, 8, 16, dtype=torch.float32)
    b_scale = torch.randn(1, 128, 16, dtype=torch.float32)
    bias = torch.randn(128, dtype=torch.bfloat16)
    residual_payload = torch.randn(1, 8, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    residual_scale = torch.randn(1, 8, 4, dtype=torch.float32)

    payload, scale = MNE_OPS._call_linear_block32_backend(
        a,
        b_t,
        a_scale,
        b_scale,
        bias,
        "bfloat16",
        False,
        True,
        False,
        residual_payload,
        residual_scale,
    )

    assert payload.shape == (1, 8, 128)
    assert scale.shape == (1, 8, 4)
    assert stub.calls == [(False, False)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for torch.compile graph tests")
def test_public_linear_block32_fused_compile_demotes_transition_epilogues(monkeypatch):
    class StubExtension:
        def __init__(self):
            self.calls: list[tuple[bool, bool]] = []

        def linear_block32_fused(
            self,
            a,
            b_t,
            a_scale_swizzled,
            b_scale_swizzled,
            bias,
            out_dtype,
            apply_relu,
            direct_fp8_output,
            fuse_bias_epilogue,
            residual_payload,
            residual_scale,
        ):
            self.calls.append((bool(direct_fp8_output), bool(fuse_bias_epilogue)))
            return MNE_OPS._linear_payload_and_scale(a, int(b_t.shape[1]))

    stub = StubExtension()
    monkeypatch.setattr(MNE_OPS, "_C", stub)
    _clear_linear_block32_retry_cache()

    a = torch.randn(1, 16, 128, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    b_t = torch.randn(1, 512, 128, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.randn(1, 16, 4, device="cuda", dtype=torch.float32)
    b_scale = torch.randn(1, 512, 4, device="cuda", dtype=torch.float32)
    bias = torch.randn(512, device="cuda", dtype=torch.bfloat16)

    def fn(a, b_t, a_scale, b_scale, bias):
        return MNE_OPS.linear_block32_fused(
            a,
            b_t,
            a_scale,
            b_scale,
            bias,
            "bfloat16",
            True,
            False,
            True,
            None,
            None,
        )

    _assert_no_graph_breaks(fn, a, b_t, a_scale, b_scale, bias)
    payload, scale = fn(a, b_t, a_scale, b_scale, bias)
    assert payload.shape == (1, 16, 512)
    assert scale.shape == (1, 16, 16)
    assert stub.calls
    assert all(call == (False, False) for call in stub.calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for torch.compile graph tests")
def test_plain_tri_mul_xbdnn_compiles_without_graph_breaks():
    x_bdnn = torch.randn((1, 128, 32, 32), device="cuda", dtype=torch.bfloat16)
    _assert_no_graph_breaks(lambda tensor: PLAIN_MODULE.tri_mul_xbdnn(tensor, out_dtype=torch.bfloat16), x_bdnn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for torch.compile graph tests")
def test_plain_tri_mul_fp8_cublaslt_compiles_without_graph_breaks():
    x_bdnn = torch.randn((1, 128, 32, 32), device="cuda", dtype=torch.bfloat16)
    _assert_no_graph_breaks(lambda tensor: PLAIN_MODULE.tri_mul_fp8_cublaslt(tensor, out_dtype=torch.bfloat16), x_bdnn)


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native fused linear compile regression",
)
def test_native_linear_forward_quantized_compile_retries_fused_epilogue():
    _clear_linear_block32_retry_cache()
    linear = torch.nn.Linear(128, 512, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(linear, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )

    def fn(inp):
        return PLAIN_MODULE.native_linear_forward_quantized(linear, inp, apply_relu=True, fuse_bias_epilogue=True)

    dynamo.reset()
    explain = dynamo.explain(fn)(x)
    assert explain.graph_break_count == 0
    assert explain.graph_count == 1

    y = fn(x)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 32, 32, 512)
    assert y.scale.shape == (1, 32, 32, 16)


def test_configure_linear_precision_registers_fp8_buffers():
    block = PLAIN_MODULE.TriangularUpdate(dim=16, linear_precision="bf16").to(dtype=torch.bfloat16)
    block.set_linear_precision("fp8")
    assert block.linear_precision == "fp8"
    assert block.pi.weight_fp8.dtype == torch.float8_e4m3fn
    assert block.pi.scale_w.dtype == torch.float32


def test_configure_linear_precision_transition_buffers():
    block = PLAIN_MODULE.TransitionUpdate(dim=16, hidden=32, linear_precision="bf16").to(dtype=torch.bfloat16)
    block.set_linear_precision("fp8", include_transition=True)
    assert block.fc1.weight_fp8.dtype == torch.float8_e4m3fn
    assert block.fc2.scale_w.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused tri-output smoke")
def test_fp8_tri_outputs_to_carrier_shapes():
    x1 = torch.randn(32, 32, 32, device="cuda", dtype=torch.float16)
    x2 = torch.randn(32, 32, 32, device="cuda", dtype=torch.float16)
    payload, scale = PLAIN_MODULE.fp8_tri_outputs_to_carrier(x1, x2, batch=1, scale_dtype=torch.float32)
    assert payload.shape == (1, 32, 32, 64)
    assert scale.shape == (1, 32, 32, 1)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused linear-output requant smoke")
def test_fp8_requantize_rows_shapes():
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    payload, scale = PLAIN_MODULE.fp8_requantize_rows(x, scale_dtype=torch.float32)
    assert payload.shape == x.shape
    assert scale.shape == (16, 1)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for block32 bias-requant smoke")
def test_fp8_requantize_block32_with_bias_shapes():
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    payload, scale = PLAIN_MODULE.fp8_requantize_block32(x, bias=bias, scale_dtype=torch.float32)
    assert payload.shape == x.shape
    assert scale.shape == (8, 4)
    assert payload.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native fused linear smoke",
)
def test_native_transition_norm_fc1_quantized_shapes():
    norm = torch.nn.LayerNorm(128, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    fc1 = torch.nn.Linear(128, 512, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(fc1, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_transition_norm_fc1_quantized(norm, fc1, x)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 512)
    assert y.scale.shape == (1, 8, 8, 16)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native fused linear smoke",
)
def test_native_linear_forward_quantized_shapes():
    linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(linear, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_linear_forward_quantized(linear, x)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 128)
    assert y.scale.shape == (1, 8, 8, 4)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or PLAIN_MODULE.minifold_native_raw is None
    or os.environ.get("MINIFOLD_TEST_EXPERIMENTAL_FC1_DIRECT") != "1",
    reason="CUDA, minifold_native_ext, and MINIFOLD_TEST_EXPERIMENTAL_FC1_DIRECT=1 are required for native fc1 direct smoke",
)
def test_native_fc1_direct_path_repeated_smoke():
    linear = torch.nn.Linear(128, 512, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(linear, enabled=True)
    linear._native_fc1_direct_enabled = True
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    for _ in range(4):
        y = PLAIN_MODULE.native_linear_forward_quantized(linear, x, apply_relu=True)
        torch.cuda.synchronize()
        assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
        assert y.payload.shape == (1, 32, 32, 512)
        assert y.scale.shape == (1, 32, 32, 16)
        assert y.payload.dtype == torch.float8_e4m3fn
        assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native fused linear smoke",
)
def test_native_linear_forward_quantized_direct_output_shapes():
    linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(linear, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_linear_forward_quantized(linear, x, direct_fp8_output=True)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 128)
    assert y.scale.shape == (1, 8, 8, 4)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native gate smoke",
)
def test_native_gate_sigmoid_mul_quantized_shapes():
    pi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    gi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(pi, enabled=True)
    PLAIN_MODULE.configure_fp8_linear_weight_(gi, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    y = PLAIN_MODULE.native_gate_sigmoid_mul_quantized(pi, gi, x)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 8, 8, 128)
    assert y.scale.shape == (1, 8, 8, 4)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native block32 pointwise smoke",
)
def test_native_block32_pointwise_shapes():
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    norm = torch.nn.LayerNorm(128, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    y = PLAIN_MODULE.native_mxfp8_layernorm_quantized(norm, x)
    z = PLAIN_MODULE.native_mxfp8_relu_quantized(y)
    added = PLAIN_MODULE.native_mxfp8_add_quantized(z, z)
    assert added.payload.shape == (1, 8, 8, 128)
    assert added.scale.shape == (1, 8, 8, 4)
    assert added.payload.dtype == torch.float8_e4m3fn
    assert added.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native block32 pointwise smoke",
)
def test_native_block32_layernorm_64_shapes():
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 8, 8, 64, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    norm = torch.nn.LayerNorm(64, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    y = PLAIN_MODULE.native_mxfp8_layernorm_quantized(norm, x)
    assert y.payload.shape == (1, 8, 8, 64)
    assert y.scale.shape == (1, 8, 8, 2)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native tri smoke",
)
def test_native_tri_mul_from_block32_quantized_shapes():
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    mask = torch.ones(1, 32, device="cuda", dtype=torch.bool)
    y = PLAIN_MODULE.native_tri_mul_from_block32_quantized(x, mask=mask)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 32, 32, 64)
    assert y.scale.shape == (1, 32, 32, 2)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native tri gate smoke",
)
def test_native_tri_mul_from_gate_quantized_shapes():
    pi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    gi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(pi, enabled=True)
    PLAIN_MODULE.configure_fp8_linear_weight_(gi, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    mask = torch.ones(1, 32, device="cuda", dtype=torch.bool)
    y = PLAIN_MODULE.native_tri_mul_from_gate_quantized(pi, gi, x, mask=mask)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 32, 32, 64)
    assert y.scale.shape == (1, 32, 32, 2)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native tri input-norm smoke",
)
def test_native_tri_mul_from_input_quantized_shapes():
    input_norm = torch.nn.LayerNorm(128, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    pi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    gi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(pi, enabled=True)
    PLAIN_MODULE.configure_fp8_linear_weight_(gi, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    mask = torch.ones(1, 32, device="cuda", dtype=torch.bool)
    y = PLAIN_MODULE.native_tri_mul_from_input_quantized(input_norm, pi, gi, x, mask=mask)
    assert isinstance(y, PLAIN_MODULE.Mxfp8PairTensor)
    assert y.payload.shape == (1, 32, 32, 64)
    assert y.scale.shape == (1, 32, 32, 2)
    assert y.payload.dtype == torch.float8_e4m3fn
    assert y.scale.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or PLAIN_MODULE.minifold_native_raw is None
    or not hasattr(PLAIN_MODULE.minifold_native_raw, "_debug_gate_sigmoid_mul_pack_to_mxfp8_reference"),
    reason="CUDA and debug gate-pack bindings are required for operand equivalence",
)
def test_debug_gate_pack_warp_matches_reference():
    torch.manual_seed(0)
    rows = 32 * 32
    lhs = torch.randn(1, rows, 128, device="cuda", dtype=torch.bfloat16)
    rhs = torch.randn(1, rows, 128, device="cuda", dtype=torch.bfloat16)
    lhs_bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    rhs_bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    mask = torch.randint(0, 2, (1, 32, 32), device="cuda", dtype=torch.bool)

    ref = PLAIN_MODULE.minifold_native_raw._debug_gate_sigmoid_mul_pack_to_mxfp8_reference(
        lhs, rhs, lhs_bias, rhs_bias, mask
    )
    warp = PLAIN_MODULE.minifold_native_raw._debug_gate_sigmoid_mul_pack_to_mxfp8_warp(
        lhs, rhs, lhs_bias, rhs_bias, mask
    )

    assert len(ref) == len(warp) == 8
    for ref_tensor, warp_tensor in zip(ref, warp):
        assert ref_tensor.dtype == warp_tensor.dtype
        assert ref_tensor.shape == warp_tensor.shape
        assert torch.equal(ref_tensor, warp_tensor)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or PLAIN_MODULE.minifold_native_raw is None
    or not hasattr(PLAIN_MODULE.minifold_native_raw, "_debug_tri_input_norm_gate_block32_reference_stages"),
    reason="CUDA and staged tri-input debug bindings are required for stage equivalence",
)
def test_debug_tri_input_stages_reference_match_warp():
    torch.manual_seed(0)
    input_norm = torch.nn.LayerNorm(128, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    pi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    gi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(pi, enabled=True)
    PLAIN_MODULE.configure_fp8_linear_weight_(gi, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 32, 32, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    mask = torch.randint(0, 2, (1, 32, 32), device="cuda", dtype=torch.bool)

    ref = PLAIN_MODULE.minifold_native_raw._debug_tri_input_norm_gate_block32_reference_stages(
        x.payload.contiguous(),
        x.scale.contiguous(),
        input_norm.weight.contiguous(),
        input_norm.bias.contiguous(),
        float(input_norm.eps),
        pi.weight_mxfp8.contiguous(),
        pi.scale_w_mxfp8_swizzled.contiguous(),
        pi.bias.contiguous(),
        gi.weight_mxfp8.contiguous(),
        gi.scale_w_mxfp8_swizzled.contiguous(),
        gi.bias.contiguous(),
        mask,
        "float16",
    )
    warp = PLAIN_MODULE.minifold_native_raw._debug_tri_input_norm_gate_block32_warp_stages(
        x.payload.contiguous(),
        x.scale.contiguous(),
        input_norm.weight.contiguous(),
        input_norm.bias.contiguous(),
        float(input_norm.eps),
        pi.weight_mxfp8.contiguous(),
        pi.scale_w_mxfp8_swizzled.contiguous(),
        pi.bias.contiguous(),
        gi.weight_mxfp8.contiguous(),
        gi.scale_w_mxfp8_swizzled.contiguous(),
        gi.bias.contiguous(),
        mask,
        "float16",
    )

    assert len(ref) == len(warp) == 16
    for ref_tensor, warp_tensor in zip(ref, warp):
        assert ref_tensor.dtype == warp_tensor.dtype
        assert ref_tensor.shape == warp_tensor.shape
        if ref_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            ref_fp32 = ref_tensor.float()
            warp_fp32 = warp_tensor.float()
            ref_nan = torch.isnan(ref_fp32)
            warp_nan = torch.isnan(warp_fp32)
            assert torch.equal(ref_nan, warp_nan)
            finite_mask = torch.isfinite(ref_fp32) & torch.isfinite(warp_fp32)
            if finite_mask.any():
                torch.testing.assert_close(ref_fp32[finite_mask], warp_fp32[finite_mask], rtol=0.0, atol=0.0)
        elif ref_tensor.is_floating_point():
            ref_nan = torch.isnan(ref_tensor)
            warp_nan = torch.isnan(warp_tensor)
            assert torch.equal(ref_nan, warp_nan)
            finite_mask = torch.isfinite(ref_tensor) & torch.isfinite(warp_tensor)
            if finite_mask.any():
                torch.testing.assert_close(ref_tensor[finite_mask], warp_tensor[finite_mask], rtol=0.0, atol=0.0)
        else:
            assert torch.equal(ref_tensor, warp_tensor)


@pytest.mark.skipif(
    not torch.cuda.is_available() or PLAIN_MODULE.minifold_native_raw is None,
    reason="CUDA and minifold_native_ext are required for native tri input-norm equivalence",
)
def test_native_tri_mul_from_input_quantized_matches_fallback(monkeypatch):
    torch.manual_seed(0)
    input_norm = torch.nn.LayerNorm(128, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    pi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    gi = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=torch.bfloat16)
    PLAIN_MODULE.configure_fp8_linear_weight_(pi, enabled=True)
    PLAIN_MODULE.configure_fp8_linear_weight_(gi, enabled=True)
    x = PLAIN_MODULE.Mxfp8PairTensor.from_tensor(
        torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16),
        scale_dtype=torch.float32,
    )
    mask = torch.randint(0, 2, (1, 256), device="cuda", dtype=torch.bool)

    y_fast = PLAIN_MODULE.native_tri_mul_from_input_quantized(input_norm, pi, gi, x, mask=mask)

    monkeypatch.delattr(PLAIN_MODULE.minifold_native_raw, "tri_input_norm_gate_block32_fused", raising=False)
    monkeypatch.delattr(PLAIN_MODULE.minifold_native_raw, "tri_gate_block32_fused", raising=False)
    y_fallback = PLAIN_MODULE.native_tri_mul_from_input_quantized(input_norm, pi, gi, x, mask=mask)

    assert y_fast.payload.shape == y_fallback.payload.shape == (1, 256, 256, 64)
    assert y_fast.scale.shape == y_fallback.scale.shape == (1, 256, 256, 2)
    assert y_fast.payload.dtype == y_fallback.payload.dtype == torch.float8_e4m3fn
    assert y_fast.scale.dtype == y_fallback.scale.dtype == torch.float32

    y_fast_bf16 = y_fast.dequantize(torch.float32)
    y_fallback_bf16 = y_fallback.dequantize(torch.float32)
    fast_nan_mask = torch.isnan(y_fast_bf16)
    fallback_nan_mask = torch.isnan(y_fallback_bf16)
    assert torch.equal(fast_nan_mask, fallback_nan_mask)
    finite_mask = torch.isfinite(y_fast_bf16) & torch.isfinite(y_fallback_bf16)
    assert finite_mask.any()
    torch.testing.assert_close(y_fast_bf16[finite_mask], y_fallback_bf16[finite_mask], rtol=0.0, atol=0.0)
