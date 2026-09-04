#!/usr/bin/env python
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

"""Minimal reproduction of FSDP2 + MXFP8 checkpoint resume crash.

Bug: After fully_shard() wraps a model with quantized_model_init (MXFP8) params,
checkpoint resume via set_state_dict crashes with:
    RuntimeError: Attempted to access the data pointer on an invalid python storage.

Root cause: set_state_dict -> model.load_state_dict -> copy_() on MXFP8Tensor
re-quantizes, allocating new internal storage. FSDP2's reset_sharded_param
(post-load hook) then calls untyped_storage().data_ptr() on the invalidated
storage. PyTorch has a "# TODO: need to support tensor subclass" comment at
the crash site (_fsdp_param.py line 892).

Fix: Wrap the data_ptr() comparison in try/except RuntimeError. When it fails,
treat as same_local_tensor=False so _sharded_param_data gets re-recorded.

Run with: torchrun --nproc_per_node=2 test_mxfp8_fsdp2_checkpoint_resume.py
"""

import argparse
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import transformer_engine.pytorch as te
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor
from torch.nn import functional as f_nn
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.optimizers import FusedAdam
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


HIDDEN = 256
FFN_HIDDEN = 1024
NUM_HEADS = 8
NUM_LAYERS = 2
SEQ_LEN = 32
BATCH = 2


def apply_reset_sharded_param_fix():
    """Monkey-patch FSDPParam.reset_sharded_param to handle QuantizedTensor.

    After checkpoint load, copy_() on MXFP8Tensor re-quantizes which can
    invalidate the old untyped_storage, causing data_ptr() to crash.
    This wraps the comparison in try/except so reset_sharded_param can
    proceed normally (re-recording _sharded_param_data).
    """

    def _patched_reset_sharded_param(self):
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
            if torch.__future__.get_swap_module_params_on_conversion():
                raise AssertionError(
                    f"Expects swap_tensors to preserve object but got {new_param} instead of {self.sharded_param}"
                )
            self.sharded_param = new_param

        local_tensor = new_param._local_tensor
        if local_tensor.is_meta:
            return

        updated_local_tensor = False
        same_local_tensor = False

        if type(self._sharded_param_data) is torch.Tensor:
            try:
                same_local_tensor = (
                    self._sharded_param_data.untyped_storage().data_ptr() > 0
                    and self._sharded_param_data.untyped_storage().data_ptr()
                    == local_tensor.untyped_storage().data_ptr()
                )
            except RuntimeError:
                # QuantizedTensor (e.g. MXFP8Tensor) can have invalid storage
                # after copy_() re-quantization.
                same_local_tensor = False

        padded_sharded_size = self.padded_sharded_param_size
        shard_dim = self.fsdp_placement.dim
        length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0

        if local_tensor.size() != padded_sharded_size and not same_local_tensor:
            if shard_dim != 0:
                raise AssertionError(f"Shard({shard_dim}) requires even sharding: {local_tensor.size()=}")
            padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
            padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(local_tensor)
            local_tensor = padded_local_tensor
            updated_local_tensor = True

        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
            updated_local_tensor = True

        if not same_local_tensor:
            self._sharded_param_data = local_tensor.view(-1)

        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")

        if updated_local_tensor:
            self.sharded_param._local_tensor = local_tensor.narrow(dim=shard_dim, start=0, length=length)
            if not self.sharded_param._local_tensor.is_contiguous():
                raise AssertionError("Expected sharded_param._local_tensor to be contiguous")

        self._sharding_spec = self.sharded_param._spec

    FSDPParam.reset_sharded_param = _patched_reset_sharded_param


def _save_custom_attrs(model):
    """Save custom attrs on QuantizedTensor params (lost during fully_shard + reset_parameters)."""
    attrs = {}
    for name, param in model.named_parameters():
        local = param._local_tensor if isinstance(param, DTensor) else param
        if isinstance(local, QuantizedTensor):
            param_attrs = {}
            for attr_name in dir(local):
                if not attr_name.startswith("_") and not callable(getattr(local, attr_name, None)):
                    try:
                        param_attrs[attr_name] = getattr(local, attr_name)
                    except Exception:
                        pass
            attrs[name] = param_attrs
    return attrs


def _restore_custom_attrs(model, attrs):
    """Restore custom attrs on QuantizedTensor params."""
    for name, param in model.named_parameters():
        if name in attrs:
            local = param._local_tensor if isinstance(param, DTensor) else param
            if isinstance(local, QuantizedTensor):
                for attr_name, attr_val in attrs[name].items():
                    try:
                        setattr(local, attr_name, attr_val)
                    except Exception:
                        pass


def build_model(recipe):
    """Build model with quantized_model_init on meta device."""
    with te.quantized_model_init(
        recipe=recipe,
        enabled=True,
        preserve_high_precision_init_val=True,
    ):
        model = torch.nn.Sequential(
            *[
                te.TransformerLayer(
                    HIDDEN,
                    FFN_HIDDEN,
                    NUM_HEADS,
                    fuse_qkv_params=True,
                    params_dtype=torch.bfloat16,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    device="meta",
                )
                for _ in range(NUM_LAYERS)
            ]
        )
    return model


def shard_model(model, mesh):
    """Apply FSDP2 sharding, then materialize meta params via reset_parameters."""
    has_meta = any(p.is_meta for p in model.parameters())
    custom_attrs = _save_custom_attrs(model)
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    if has_meta:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    _restore_custom_attrs(model, custom_attrs)
    return model


def build_and_shard(recipe, mesh, device):
    """Build model, shard, create optimizer, run one step to populate optimizer state."""
    model = build_model(recipe)
    model = shard_model(model, mesh)

    optimizer = FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float32,
    )

    # Run one training step to populate optimizer state
    x = torch.randn(SEQ_LEN, BATCH, HIDDEN, dtype=torch.bfloat16, device=device)
    target = torch.randn_like(x)
    optimizer.zero_grad(set_to_none=True)
    with te.autocast(enabled=True, recipe=recipe):
        out = model(x)
    loss = f_nn.mse_loss(out, target)
    loss.backward()
    optimizer.step()

    return model, optimizer


def run(apply_fix: bool):
    """Run the reproduction: save checkpoint, load it, verify forward pass."""
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    world_size = dist.get_world_size()
    mesh = DeviceMesh("cuda", list(range(world_size)))

    recipe = MXFP8BlockScaling()

    if apply_fix:
        apply_reset_sharded_param_fix()
        if rank == 0:
            print("Applied reset_sharded_param fix")

    # Build model, train one step, save checkpoint
    model, optimizer = build_and_shard(recipe, mesh, device)
    if rank == 0:
        print("Model built and trained for 1 step")

    # Record reference output
    x = torch.randn(SEQ_LEN, BATCH, HIDDEN, dtype=torch.bfloat16, device=device)
    with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
        ref_output = model(x).clone()
    if rank == 0:
        print(f"Reference output recorded, norm={ref_output.norm().item():.4f}")

    checkpoint_dir = "/tmp/te_test_mxfp8_fsdp2_ckpt_resume"
    if rank == 0:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    dist.barrier()

    try:
        # Save checkpoint
        model_state = {k: v for k, v in model.state_dict().items() if not k.endswith("_extra_state")}
        dcp.save({"model": model_state, "optimizer": optimizer.state_dict()}, checkpoint_id=checkpoint_dir)
        dist.barrier()
        if rank == 0:
            print(f"Checkpoint saved to {checkpoint_dir}")

        # Build fresh model
        model2, optimizer2 = build_and_shard(recipe, mesh, device)
        if rank == 0:
            print("Fresh model built, loading checkpoint...")

        # Load checkpoint — THIS IS WHERE THE CRASH HAPPENS WITHOUT THE FIX
        model2_state = {k: v for k, v in model2.state_dict().items() if not k.endswith("_extra_state")}
        state_to_load = {"model": model2_state, "optimizer": optimizer2.state_dict()}
        dcp.load(state_to_load, checkpoint_id=checkpoint_dir)
        model2.load_state_dict(state_to_load["model"], strict=False)
        optimizer2.load_state_dict(state_to_load["optimizer"])
        dist.barrier()
        if rank == 0:
            print("Checkpoint loaded successfully!")

        # Verify output matches
        with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
            loaded_output = model2(x)

        torch.testing.assert_close(
            loaded_output,
            ref_output,
            rtol=0,
            atol=0,
            msg=lambda m: f"Output mismatch after checkpoint load: {m}",
        )
        if rank == 0:
            print("Output parity verified — bitwise identical!")

    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    dist.destroy_process_group()
    if rank == 0:
        print("SUCCESS" if apply_fix else "SUCCESS (unexpected — bug may be fixed upstream)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Apply the reset_sharded_param monkey-patch fix")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    run(apply_fix=args.fix)
