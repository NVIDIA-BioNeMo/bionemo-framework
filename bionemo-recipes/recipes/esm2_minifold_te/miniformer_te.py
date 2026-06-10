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

import os
import warnings
from contextlib import nullcontext
from typing import ContextManager

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from torch import Tensor
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor

try:
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        fused_relu_backward_fp8,
        fused_relu_forward_fp8,
        fused_sigmoid_gate_backward_fp8,
        fused_sigmoid_gate_forward_fp8,
    )
except ImportError:
    def _missing_fp8_blockwise_helper(name: str):
        def _missing(*args, **kwargs):
            raise RuntimeError(
                f"{name} is unavailable in the installed TransformerEngine build on this node"
            )

        return _missing

    fused_relu_backward_fp8 = _missing_fp8_blockwise_helper("fused_relu_backward_fp8")
    fused_relu_forward_fp8 = _missing_fp8_blockwise_helper("fused_relu_forward_fp8")
    fused_sigmoid_gate_backward_fp8 = _missing_fp8_blockwise_helper("fused_sigmoid_gate_backward_fp8")
    fused_sigmoid_gate_forward_fp8 = _missing_fp8_blockwise_helper("fused_sigmoid_gate_forward_fp8")

from minifold_utils import init
from flash_trimul import flash_trimul
from quantization import ComponentPrecisionConfig
from te_utils import (
    quantize_tensor_for_fp8_save,
    restore_fp8_tensors_from_backward,
    save_fp8_tensors_for_backward,
    saved_tensor_fp8_context,
    te_layernorm_nd,
    te_linear_nd,
    tri_mul,
    tri_mul_bmm_bdnn,
    tri_mul_fp8_bdnn,
    tri_mul_fp8_cublaslt_bdnn,
    tri_mul_fp8_fused_bdnn,
    tri_mul_fp8_grouped_bdnn,
    tri_mul_xbdnn,
)


class _TransitionUpdateFP8SaveFn(Function):
    """Custom autograd wrapper that saves TransitionUpdateTE state in FP8 when possible."""

    @staticmethod
    def forward(ctx, x, norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, mod):
        del norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias
        ctx.mod = mod
        ctx.fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if ctx.fp8_enabled else None
        ctx.x_dtype = x.dtype
        ctx.debug_memory = bool(
            getattr(getattr(mod, "_component_precision", None), "ffn_fused_subgraph_debug_memory", False)
        )

        x_norm = te_layernorm_nd(mod.norm, x)
        hidden = te_linear_nd(mod.fc1, x_norm, fp8_output=ctx.fp8_enabled)
        if isinstance(hidden, QuantizedTensor):
            hidden = hidden.dequantize(dtype=x.dtype)
        relu = F.relu(hidden)
        out = te_linear_nd(mod.fc2, relu)

        x_saved = quantize_tensor_for_fp8_save(x)
        save_fp8_tensors_for_backward(ctx, x_saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        def _debug(tag: str):
            if not ctx.debug_memory or not torch.cuda.is_available():
                return
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 2**30
            reserv = torch.cuda.memory_reserved() / 2**30
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[TransitionFP8Backward] {tag}: alloc={alloc:.4f} GiB reserved={reserv:.4f} GiB peak={peak:.4f} GiB")

        mod = ctx.mod
        _debug("start")
        restored = restore_fp8_tensors_from_backward(ctx)
        (x_saved,) = restored
        _debug("restored_saved")
        x = x_saved.dequantize(dtype=ctx.x_dtype) if isinstance(x_saved, QuantizedTensor) else x_saved
        del restored, x_saved
        _debug("dequantized_saved")

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            with (te.autocast(enabled=True, recipe=ctx.fp8_recipe) if ctx.fp8_enabled else nullcontext()):
                x_norm = te_layernorm_nd(mod.norm, x)
                hidden = te_linear_nd(mod.fc1, x_norm)
                relu_re = F.relu(hidden)
                out_fc2 = te_linear_nd(mod.fc2, relu_re)
            _debug("after_fc2_recompute")
            grad_relu, grad_fc2_w, grad_fc2_b = torch.autograd.grad(
                out_fc2,
                [relu_re, mod.fc2.weight, mod.fc2.bias],
                grad_out,
                allow_unused=True,
            )
            del out_fc2, hidden, x_norm
            _debug("after_fc2_grad")

            with (te.autocast(enabled=True, recipe=ctx.fp8_recipe) if ctx.fp8_enabled else nullcontext()):
                x_norm = te_layernorm_nd(mod.norm, x)
                hidden = te_linear_nd(mod.fc1, x_norm)
                relu_re_2 = F.relu(hidden)
            _debug("after_fc1_recompute")
            grad_x, grad_norm_w, grad_norm_b, grad_fc1_w, grad_fc1_b = torch.autograd.grad(
                relu_re_2,
                [x, mod.norm.weight, mod.norm.bias, mod.fc1.weight, mod.fc1.bias],
                grad_relu,
                allow_unused=True,
            )
        del relu_re, relu_re_2, hidden, x_norm, grad_relu, x
        _debug("after_fc1_grad")
        return grad_x, grad_norm_w, grad_norm_b, grad_fc1_w, grad_fc1_b, grad_fc2_w, grad_fc2_b, None


class _GatedChainToFloat8Func(Function):
    """Autograd bridge for the triangular sigmoid->mul chain with FP8 output."""

    @staticmethod
    def forward(ctx, proj_out, gate_logits, mask, quantize_output: bool):
        proj_dtype = proj_out.dtype if isinstance(proj_out, torch.Tensor) else None
        gate_dtype = gate_logits.dtype if isinstance(gate_logits, torch.Tensor) else None

        use_fused_fp8 = (
            quantize_output
            and isinstance(proj_out, Float8BlockwiseQTensor)
            and isinstance(gate_logits, Float8BlockwiseQTensor)
        )
        if use_fused_fp8:
            try:
                out_q, saved_g_q = fused_sigmoid_gate_forward_fp8(
                    proj_out,
                    gate_logits,
                    mask,
                )
                ctx.save_for_backward(
                    proj_out,
                    saved_g_q,
                    mask if mask is not None else torch.tensor([], device=out_q.device),
                )
                ctx.has_mask = mask is not None
                ctx.proj_dtype = proj_dtype
                ctx.gate_dtype = gate_dtype
                ctx.quantize_output = quantize_output
                ctx.use_fused_fp8 = True
                return out_q.requires_grad_(proj_out.requires_grad or gate_logits.requires_grad)
            except Exception:
                pass

        proj_hp = proj_out.dequantize(dtype=proj_dtype) if isinstance(proj_out, QuantizedTensor) else proj_out
        gate_hp = gate_logits.dequantize(dtype=gate_dtype) if isinstance(gate_logits, QuantizedTensor) else gate_logits

        gate_sigmoid = torch.sigmoid(gate_hp)
        out = proj_hp * gate_sigmoid
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        ctx.save_for_backward(proj_hp, gate_sigmoid, mask if mask is not None else torch.tensor([], device=out.device))
        ctx.has_mask = mask is not None
        ctx.proj_dtype = proj_dtype
        ctx.gate_dtype = gate_dtype
        ctx.quantize_output = quantize_output
        ctx.use_fused_fp8 = False
        if not quantize_output:
            return out

        quantizer_src = proj_out if isinstance(proj_out, QuantizedTensor) else gate_logits
        if not isinstance(quantizer_src, QuantizedTensor):
            return out
        quantizer = quantizer_src._get_quantizer().copy()
        quantizer.set_usage(rowwise=True, columnwise=True)
        out_q = quantizer.quantize(out, dtype=quantizer_src.dtype)
        return out_q.requires_grad_(out.requires_grad)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.use_fused_fp8:
            proj_q, saved_g_q, mask = ctx.saved_tensors
            grad_proj, grad_gate = fused_sigmoid_gate_backward_fp8(
                proj_q,
                saved_g_q,
                grad_out,
                mask if ctx.has_mask else None,
                proj_dtype=ctx.proj_dtype,
                gate_dtype=ctx.gate_dtype,
            )
            return grad_proj, grad_gate, None, None, None

        proj_hp, gate_sigmoid, mask = ctx.saved_tensors

        if isinstance(grad_out, QuantizedTensor):
            grad_out = grad_out.dequantize(dtype=ctx.proj_dtype or ctx.gate_dtype)

        target_dtype = ctx.proj_dtype or ctx.gate_dtype
        if target_dtype is not None and grad_out.dtype != target_dtype:
            grad_out = grad_out.to(target_dtype)

        if ctx.has_mask:
            grad_out = grad_out * mask.unsqueeze(-1)

        grad_proj = grad_out * gate_sigmoid
        grad_gate = grad_out * proj_hp * gate_sigmoid * (1 - gate_sigmoid)

        if ctx.proj_dtype is not None and grad_proj.dtype != ctx.proj_dtype:
            grad_proj = grad_proj.to(ctx.proj_dtype)
        if ctx.gate_dtype is not None and grad_gate.dtype != ctx.gate_dtype:
            grad_gate = grad_gate.to(ctx.gate_dtype)

        return grad_proj, grad_gate, None, None, None


class _NativeFP8ReluFunc(Function):
    """ReLU on raw Float8BlockwiseQTensor payloads with FP8 saved state."""

    @staticmethod
    def _positive_mask(tensor: QuantizedTensor) -> torch.Tensor:
        if isinstance(tensor, QuantizedTensor) and getattr(tensor, "_rowwise_data", None) is not None:
            data = tensor._rowwise_data
            return torch.bitwise_and(data, 0x80).eq(0) & data.ne(0)
        if isinstance(tensor, QuantizedTensor) and getattr(tensor, "_columnwise_data", None) is not None:
            data = tensor._columnwise_data
            mask = torch.bitwise_and(data, 0x80).eq(0) & data.ne(0)
            return mask.permute(*range(1, mask.dim()), 0).contiguous()
        raise RuntimeError("Native FP8 ReLU requires Float8BlockwiseQTensor data buffers")

    @staticmethod
    def forward(ctx, tensor, quantize_output: bool):
        input_dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else None
        ctx.input_dtype = input_dtype
        if not isinstance(tensor, QuantizedTensor) or not quantize_output:
            out = F.relu(tensor)
            ctx.save_for_backward(out)
            ctx.quantized = False
            return out
        out_q = fused_relu_forward_fp8(tensor)
        ctx.save_for_backward(out_q)
        ctx.quantized = True
        return out_q.requires_grad_(tensor.requires_grad)

    @staticmethod
    def backward(ctx, grad_out):
        (saved_out,) = ctx.saved_tensors
        if isinstance(grad_out, QuantizedTensor):
            grad_out = grad_out.dequantize(dtype=ctx.input_dtype)
        if ctx.input_dtype is not None and grad_out.dtype != ctx.input_dtype:
            grad_out = grad_out.to(ctx.input_dtype)
        if ctx.quantized:
            grad_input = fused_relu_backward_fp8(
                saved_out._rowwise_data,
                grad_out,
                input_dtype=ctx.input_dtype,
            )
        else:
            grad_input = grad_out * (saved_out > 0).to(dtype=grad_out.dtype)
        if ctx.input_dtype is not None and grad_input.dtype != ctx.input_dtype:
            grad_input = grad_input.to(ctx.input_dtype)
        return grad_input, None


class TransitionUpdateTE(nn.Module):
    """TE version of TransitionUpdate: two-layer MLP with residual connection.

    Replaces raw nn.Parameter + F.linear with te.LayerNorm + te.Linear modules.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        params_dtype: torch.dtype = torch.float32,
        component_precision: ComponentPrecisionConfig | None = None,
    ):
        super().__init__()
        self._component_precision = component_precision
        self.norm = te.LayerNorm(dim, eps=1e-5, params_dtype=params_dtype)
        self.fc1 = te.Linear(dim, hidden, params_dtype=params_dtype)
        self.fc2 = te.Linear(hidden, dim, params_dtype=params_dtype)

        # Match original initialization
        init.bias_init_one_(self.norm.weight)
        init.bias_init_zero_(self.norm.bias)
        init.he_normal_init_(self.fc1.weight)
        init.bias_init_zero_(self.fc1.bias)
        init.final_init_(self.fc2.weight)
        init.bias_init_zero_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        cp = self._component_precision
        ctx = cp.get_context("ffn") if cp else nullcontext()
        if cp and cp.ffn_fused_subgraph_fp8:
            with ctx:
                return _TransitionUpdateFP8SaveFn.apply(
                    x,
                    self.norm.weight,
                    self.norm.bias,
                    self.fc1.weight,
                    self.fc1.bias,
                    self.fc2.weight,
                    self.fc2.bias,
                    self,
                )
        save_ctx = saved_tensor_fp8_context(
            enabled=bool(cp and cp.ffn_saved_tensors_fp8),
            include_bf16=True,
            include_fp32=bool(cp and cp.saved_tensors_fp8_include_fp32),
        )
        with save_ctx:
            with ctx:
                keep_fp8_linear_outputs = FP8GlobalStateManager.is_fp8_enabled()
                x = te_layernorm_nd(self.norm, x)
                x = te_linear_nd(self.fc1, x, fp8_output=keep_fp8_linear_outputs)
                use_native_fp8_relu = bool(
                    cp and cp.ffn_relu_native_fp8 and keep_fp8_linear_outputs
                )
                if use_native_fp8_relu:
                    x = _NativeFP8ReluFunc.apply(x, True)
                else:
                    x = F.relu(x)
                x = te_linear_nd(self.fc2, x)
        return x


class TriangularUpdateTE(nn.Module):
    """TE version of TriangularUpdate.

    Replaces raw nn.Parameter + F.linear/F.layer_norm with te.LayerNorm + te.Linear.
    The einsum triangular multiplication operations remain in FP32.
    """

    def __init__(
        self,
        dim: int = 128,
        params_dtype: torch.dtype = torch.float32,
        component_precision: ComponentPrecisionConfig | None = None,
    ):
        super().__init__()
        self._component_precision = component_precision

        # Input gating: LayerNorm + two parallel linears (projection and gate)
        self.input_norm = te.LayerNorm(dim, eps=1e-5, params_dtype=params_dtype)
        self.pi = te.Linear(dim, dim, params_dtype=params_dtype)  # input projection
        self.gi = te.Linear(dim, dim, params_dtype=params_dtype)  # input gate (sigmoid)

        # Output gating: LayerNorm + two parallel linears
        self.output_norm = te.LayerNorm(dim // 2, eps=1e-5, params_dtype=params_dtype)
        self.po = te.Linear(dim // 2, dim, params_dtype=params_dtype)  # output projection
        self.go = te.Linear(dim // 2, dim, params_dtype=params_dtype)  # output gate (sigmoid)

        # Match original initialization
        init.bias_init_one_(self.input_norm.weight)
        init.bias_init_zero_(self.input_norm.bias)

        init.lecun_normal_init_(self.pi.weight)
        init.bias_init_zero_(self.pi.bias)
        init.gating_init_(self.gi.weight)
        init.bias_init_one_(self.gi.bias)

        init.bias_init_one_(self.output_norm.weight)
        init.bias_init_zero_(self.output_norm.bias)

        init.final_init_(self.po.weight)
        init.bias_init_zero_(self.po.bias)
        init.gating_init_(self.go.weight)
        init.bias_init_one_(self.go.bias)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).
            mask: Mask tensor of shape (B, N, N).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        cp = self._component_precision

        def _proj_ctx():
            return cp.get_context("tri_proj") if cp else nullcontext()

        def _gate_ctx():
            return cp.get_context("tri_gate") if cp else nullcontext()

        # TE can preserve quantized outputs across the linear boundary itself, but
        # sigmoid / elementwise ops still force a higher-precision break.
        keep_fp8_linear_outputs = FP8GlobalStateManager.is_fp8_enabled()

        save_ctx = saved_tensor_fp8_context(
            enabled=bool(cp and cp.tri_saved_tensors_fp8),
            include_bf16=True,
            include_fp32=bool(cp and cp.saved_tensors_fp8_include_fp32),
        )
        with save_ctx:
            x = self.project_and_gate(x, mask)
            x = self.tri_mul_and_output(x, mask)

        return x

    def project_and_gate(self, x: Tensor, mask: Tensor) -> Tensor:
        cp = self._component_precision

        def _proj_ctx():
            return cp.get_context("tri_proj") if cp else nullcontext()

        def _gate_ctx():
            return cp.get_context("tri_gate") if cp else nullcontext()

        keep_fp8_linear_outputs = FP8GlobalStateManager.is_fp8_enabled()
        x = te_layernorm_nd(self.input_norm, x)
        with _proj_ctx():
            pi_out = te_linear_nd(self.pi, x, fp8_output=keep_fp8_linear_outputs)
        with _gate_ctx():
            gi_logits = te_linear_nd(self.gi, x, fp8_output=keep_fp8_linear_outputs)
        tri_impl = cp.tri_impl if cp else os.environ.get("BIONEMO_TRI_MUL_IMPL", "bmm")
        if tri_impl == "flash_trimul":
            return (pi_out, gi_logits)
        use_gating_chain = bool(cp and cp.tri_gating_chain_fp8 and keep_fp8_linear_outputs)
        if use_gating_chain:
            return _GatedChainToFloat8Func.apply(
                pi_out,
                gi_logits,
                mask,
                True,
            )
        return pi_out * torch.sigmoid(gi_logits) * mask.unsqueeze(-1)

    def tri_mul_and_output(self, x: Tensor | tuple[Tensor, Tensor], mask: Tensor) -> Tensor:
        cp = self._component_precision

        def _proj_ctx():
            return cp.get_context("tri_proj") if cp else nullcontext()

        def _gate_ctx():
            return cp.get_context("tri_gate") if cp else nullcontext()

        keep_fp8_linear_outputs = FP8GlobalStateManager.is_fp8_enabled()

        tri_impl = cp.tri_impl if cp else os.environ.get("BIONEMO_TRI_MUL_IMPL", "bmm")
        tri_mode = cp.tri_einsum if cp else "off"
        use_fp32 = tri_mode == "off"
        if tri_impl == "flash_trimul":
            if not isinstance(x, tuple) or len(x) != 2:
                raise TypeError("flash_trimul expects project_and_gate to return (proj, gate_logits)")
            proj, gate_logits = x
            proj_in = proj.float() if use_fp32 else proj
            gate_in = gate_logits.float() if use_fp32 else gate_logits
            x = flash_trimul(proj_in, gate_in, mask)
        elif tri_impl in {"bmm", "cublas_xbdnn", "fp8_bmm", "fp8_cublaslt", "fp8_grouped", "fused_fp8"}:
            x_in = x.float() if use_fp32 else x
            x_bdnn = x_in.permute(0, 3, 1, 2).contiguous()
            if tri_impl == "cublas_xbdnn":
                x = tri_mul_xbdnn(x_bdnn, out_dtype=x_in.dtype)
            elif tri_impl == "fp8_bmm":
                x = tri_mul_fp8_bdnn(x_bdnn, out_dtype=x_in.dtype)
            elif tri_impl == "fp8_cublaslt":
                x = tri_mul_fp8_cublaslt_bdnn(x_bdnn, out_dtype=x_in.dtype)
            elif tri_impl == "fused_fp8":
                x = tri_mul_fp8_fused_bdnn(x_bdnn, out_dtype=x_in.dtype)
            elif tri_impl == "fp8_grouped":
                x = tri_mul_fp8_grouped_bdnn(x_bdnn, out_dtype=x_in.dtype)
            else:
                a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
                x1 = tri_mul_bmm_bdnn(a1, b1, k_dim=2)
                x2 = tri_mul_bmm_bdnn(a2, b2, k_dim=1)
                x = torch.cat([x1, x2], dim=-1)
        else:
            x_in = x.float() if use_fp32 else x
            a1, b1, a2, b2 = torch.chunk(x_in, 4, dim=-1)
            if tri_impl == "fused":
                a1 = a1.contiguous()
                b1 = b1.contiguous()
                a2 = a2.contiguous()
                b2 = b2.contiguous()
            x1 = tri_mul(a1, b1, k_dim=2, mode=tri_mode, impl=tri_impl)
            x2 = tri_mul(a2, b2, k_dim=1, mode=tri_mode, impl=tri_impl)
            x = torch.cat([x1, x2], dim=-1)
        if use_fp32:
            x = x.to(mask.dtype if mask.is_floating_point() else torch.float32)

        x = te_layernorm_nd(self.output_norm, x)
        with _proj_ctx():
            po_out = te_linear_nd(self.po, x, fp8_output=keep_fp8_linear_outputs)
        with _gate_ctx():
            go_logits = te_linear_nd(self.go, x, fp8_output=keep_fp8_linear_outputs)
        use_gating_chain = bool(cp and cp.tri_gating_chain_fp8 and keep_fp8_linear_outputs)
        if use_gating_chain:
            return _GatedChainToFloat8Func.apply(
                po_out,
                go_logits,
                None,
                True,
            )
        return po_out * torch.sigmoid(go_logits)


class BlockTE(nn.Module):
    """TE version of a MiniFormer block: TriangularUpdate + TransitionUpdate."""

    def __init__(
        self,
        dim: int = 128,
        params_dtype: torch.dtype = torch.float32,
        component_precision: ComponentPrecisionConfig | None = None,
    ):
        super().__init__()
        self.triangular = TriangularUpdateTE(dim, params_dtype=params_dtype, component_precision=component_precision)
        self.transition = TransitionUpdateTE(
            dim, dim * 4, params_dtype=params_dtype, component_precision=component_precision
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).
            mask: Mask tensor of shape (B, N, N).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        cp = self.transition._component_precision
        if (
            cp
            and cp.tri_prefix_checkpoint_reentrant
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            tri_prefix_checkpoint_save_ctx = saved_tensor_fp8_context(
                enabled=bool(cp.tri_saved_tensors_fp8),
                include_bf16=True,
                include_fp32=bool(cp.saved_tensors_fp8_include_fp32),
                include_leaf_tensors=True,
            )
            with tri_prefix_checkpoint_save_ctx:
                tri_in = checkpoint(self.triangular.project_and_gate, x, mask, use_reentrant=True)
            tri_save_ctx = saved_tensor_fp8_context(
                enabled=bool(cp.tri_saved_tensors_fp8),
                include_bf16=True,
                include_fp32=bool(cp.saved_tensors_fp8_include_fp32),
            )
            with tri_save_ctx:
                x = x + self.triangular.tri_mul_and_output(tri_in, mask)
        elif (
            cp
            and cp.tri_checkpoint_reentrant
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            tri_checkpoint_save_ctx = saved_tensor_fp8_context(
                enabled=bool(cp.tri_saved_tensors_fp8),
                include_bf16=True,
                include_fp32=bool(cp.saved_tensors_fp8_include_fp32),
                include_leaf_tensors=True,
            )
            with tri_checkpoint_save_ctx:
                x = x + checkpoint(self.triangular, x, mask, use_reentrant=True)
        else:
            x = x + self.triangular(x, mask)
        if (
            cp
            and cp.ffn_checkpoint_reentrant
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        ):
            checkpoint_save_ctx = saved_tensor_fp8_context(
                enabled=bool(cp.ffn_saved_tensors_fp8),
                include_bf16=True,
                include_fp32=bool(cp.saved_tensors_fp8_include_fp32),
                include_leaf_tensors=True,
            )
            with checkpoint_save_ctx:
                x = x + checkpoint(self.transition, x, use_reentrant=True)
        else:
            x = x + self.transition(x)
        return x


class MiniFormerTE(nn.Module):
    """TE version of the MiniFormer module with optional per-block FP8/FP4 precision."""

    def __init__(
        self,
        dim: int = 128,
        blocks: int = 48,
        params_dtype: torch.dtype = torch.float32,
        block_precision: list[str | None] | None = None,
        fp8_recipe=None,
        fp4_recipe=None,
        component_precision: ComponentPrecisionConfig | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BlockTE(dim, params_dtype=params_dtype, component_precision=component_precision) for _ in range(blocks)]
        )
        self._block_precision = block_precision
        self._fp8_recipe = fp8_recipe
        self._fp4_recipe = fp4_recipe

        if block_precision is not None and len(block_precision) != blocks:
            raise ValueError(f"block_precision length ({len(block_precision)}) must match number of blocks ({blocks})")

    def get_autocast_context(self, block_number: int | None, outer: bool = False) -> ContextManager:
        """Return the appropriate TE autocast context manager for a given block.

        Args:
            block_number: The 0-indexed block number.
            outer: Whether to return a global te.autocast() context to wrap the entire block stack.
        """
        if self._block_precision is None:
            return nullcontext()

        if outer:
            if "fp8" not in self._block_precision:
                return nullcontext()
            if self._fp8_recipe is None:
                warnings.warn("No FP8 recipe provided, using default recipe.", UserWarning)
            return te.autocast(enabled=True, recipe=self._fp8_recipe)

        precision = self._block_precision[block_number]
        recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)

        if precision == "fp8":
            if recipe is None:
                warnings.warn("No FP8 recipe provided, using default recipe.", UserWarning)
            return te.autocast(enabled=True, recipe=recipe)
        if precision == "fp4":
            if recipe is None:
                raise RuntimeError("No FP4 recipe provided, but block precision is set to FP4.")
            return te.autocast(enabled=True, recipe=recipe)
        return te.autocast(enabled=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, N, D).
            mask: Mask tensor of shape (B, N, N).

        Returns:
            Output tensor of shape (B, N, N, D).
        """
        with self.get_autocast_context(None, outer=True):
            for block_idx, block in enumerate(self.blocks):
                with self.get_autocast_context(block_idx):
                    x = block(x, mask)
        return x
