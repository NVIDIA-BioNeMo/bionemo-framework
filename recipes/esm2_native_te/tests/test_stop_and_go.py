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

"""Stop and go tests for the ESM2 Native TE recipe."""

import os
import shutil
from dataclasses import dataclass

import torch
from torch.optim import AdamW

from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp
from dataset import create_bshd_dataloader
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM
from scheduler import get_linear_schedule_with_warmup


@dataclass
class MockSingleProcessDistributedConfig:
    rank: int
    local_rank: int
    world_size: int

    def is_main_process(self):
        return self.rank == 0


def test_stop_and_go_checkpointing_and_dataloader_restoration_single_gpu(tmp_path):
    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Setup the dataloader
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockSingleProcessDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    # Setup the model
    config = NVEsmConfig.from_pretrained("model_configs/nvidia/esm2_t6_8M_UR50D", dtype=torch.bfloat16)
    model = NVEsmForMaskedLM(config)

    # The huggingface model has a contact head that we don't use in masked language pre-training, so we delete it to
    # avoid errors with unused parameters.
    base = model.model if hasattr(model, "model") else model.esm
    try:
        del base.contact_head
    except AttributeError:
        pass

    # Create optimizer.
    adamw_kwargs = {"lr": 4e-4, "fused": True, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.01}
    lr_scheduler_kwargs = {"num_warmup_steps": 2_000, "num_training_steps": 500_000}

    optimizer = AdamW(model.parameters(), **adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)

    device = torch.device(f"cuda:{dist_config.local_rank}")
    model = model.to(device=device)

    step5_path_reference = f"{tmp_path}step_5"
    step10_path_reference = f"{tmp_path}step_10"
    step5_path_reloaded = f"{tmp_path}step_5_reloaded"
    if os.path.exists(step5_path_reference):
        shutil.rmtree(step5_path_reference)
    if os.path.exists(step10_path_reference):
        shutil.rmtree(step10_path_reference)
    if os.path.exists(step5_path_reloaded):
        shutil.rmtree(step5_path_reloaded)
    os.makedirs(step5_path_reference, exist_ok=True)
    os.makedirs(step10_path_reference, exist_ok=True)
    os.makedirs(step5_path_reloaded, exist_ok=True)

    # Train for 10 steps
    model.train()
    for step, batch in enumerate(reference_dataloader):
        batch["labels"] = batch["input_ids"].clone()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)

        # Backward pass.
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}

        # Step optimizer.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step == 5:
            save_checkpoint_ddp(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=step5_path_reference,
                step=step,
                dist_config=dist_config,
                dataloader=reference_dataloader,
                epoch=0,
            )
        if step == 9:
            break

    # Now save the results after 10 steps
    save_checkpoint_ddp(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ckpt_path=step10_path_reference,
        step=step,
        dist_config=dist_config,
        dataloader=None,
        epoch=0,
    )
    torch.save(logits.cpu(), f"{step10_path_reference}_logits.pt")
    torch.save(loss.cpu(), f"{step10_path_reference}_loss.pt")
    torch.save(batch, f"{step10_path_reference}_batch.pt")
    torch.save(grads, f"{step10_path_reference}_grads.pt")
    # Create fresh model, optimizer, scheduler for the resume test
    config = NVEsmConfig.from_pretrained("model_configs/nvidia/esm2_t6_8M_UR50D", dtype=torch.bfloat16)
    resumed_model = NVEsmForMaskedLM(config)

    resumed_base = resumed_model.model if hasattr(resumed_model, "model") else resumed_model.esm
    try:
        del resumed_base.contact_head
    except AttributeError:
        pass

    resumed_model = resumed_model.to(device=device)
    resumed_optimizer = AdamW(resumed_model.parameters(), **adamw_kwargs)
    resumed_scheduler = get_linear_schedule_with_warmup(resumed_optimizer, **lr_scheduler_kwargs)

    # Now make a dataloader brand new and restore the state?
    new_dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    # Load checkpoint from step 5 into the fresh model
    resumed_model, resumed_optimizer, resumed_scheduler, new_dataloader, _, _ = load_checkpoint_ddp(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        ckpt_path=step5_path_reference,
        dist_config=dist_config,
        dataloader=new_dataloader,
    )

    # Now train for 3 more steps. Which are like training step 6-9 of the reference dataloader.
    resumed_model.train()
    for step, batch in enumerate(new_dataloader):
        batch["labels"] = batch["input_ids"].clone()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = resumed_model(**batch)

        # Backward pass.
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        resumed_grads = {name: p.grad for name, p in resumed_model.named_parameters() if p.grad is not None}

        # Step optimizer.
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_optimizer.zero_grad()
        if step == 3:
            break

    # Now save the results after 5 steps from the new dataloader. Which should match 10 steps of the reference dataloader.
    torch.save(logits.cpu(), f"{step5_path_reloaded}_logits.pt")
    torch.save(loss.cpu(), f"{step5_path_reloaded}_loss.pt")
    torch.save(batch, f"{step5_path_reloaded}_batch.pt")
    torch.save(resumed_grads, f"{step5_path_reloaded}_grads.pt")
    save_checkpoint_ddp(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        ckpt_path=step5_path_reloaded,
        step=step,
        dist_config=dist_config,
        dataloader=new_dataloader,
        epoch=0,
    )

    # Let's compare the batches now.
    reference_batch_step_10 = torch.load(f"{step10_path_reference}_batch.pt")["input_ids"]
    reloaded_batch_step_5 = torch.load(f"{step5_path_reloaded}_batch.pt")["input_ids"]
    assert torch.equal(reference_batch_step_10, reloaded_batch_step_5), (
        "Final batches don't match - dataloader state restoration may have failed"
    )

    # Let's compare the losses now.
    reference_loss_step_10 = torch.load(f"{step10_path_reference}_loss.pt")
    reloaded_loss_step_5 = torch.load(f"{step5_path_reloaded}_loss.pt")
    loss_diff = abs(reference_loss_step_10 - reloaded_loss_step_5).item()
    assert torch.allclose(reference_loss_step_10, reloaded_loss_step_5, rtol=2e-2, atol=1e-3), (
        f"Losses don't match - abs diff: {loss_diff:.6f} (reference={reference_loss_step_10.item():.6f}, reloaded={reloaded_loss_step_5.item():.6f})"
    )

    # Let's compare logits now (using allclose for floating point tolerance)
    reference_logits_step_10 = torch.load(f"{step10_path_reference}_logits.pt")
    reloaded_logits_step_5 = torch.load(f"{step5_path_reloaded}_logits.pt")

    # Calculate element-wise differences for debugging
    logit_diff = (reference_logits_step_10 - reloaded_logits_step_5).abs()
    max_diff = logit_diff.max().item()
    mean_diff = logit_diff.mean().item()

    # Find location of max difference
    max_idx = logit_diff.argmax()
    max_idx_tuple = torch.unravel_index(max_idx, logit_diff.shape)
    ref_val = reference_logits_step_10.flatten()[max_idx].item()
    reload_val = reloaded_logits_step_5.flatten()[max_idx].item()

    # BF16 tolerance: max diff of ~0.017 is normal for BF16 after 10 training steps
    # Using atol=0.02 to account for BF16 precision limitations
    assert torch.allclose(reference_logits_step_10, reloaded_logits_step_5, rtol=1e-2, atol=2.0e-2), (
        f"Logits don't match - max abs diff: {max_diff:.6f}, mean abs diff: {mean_diff:.6f}\n"
        f"Max diff at position {max_idx_tuple}: reference={ref_val:.6f}, reloaded={reload_val:.6f}"
    )

    # Now ensure the schedulers match up
    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_scheduler.base_lrs == scheduler.base_lrs
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()

    reference_grads_step_10 = torch.load(f"{step10_path_reference}_grads.pt")
    reloaded_grads_step_5 = torch.load(f"{step5_path_reloaded}_grads.pt")
    torch.testing.assert_close(reference_grads_step_10, reloaded_grads_step_5, atol=1e-2, rtol=2e-2)

    shutil.rmtree(step5_path_reference)
    shutil.rmtree(step10_path_reference)
    shutil.rmtree(step5_path_reloaded)
