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


"""Demonstration of LoRA fine-tuning of ESM-2 with Transformer Engine and PEFT using DDP."""

import logging
from pathlib import Path

import hydra
import peft
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
)

from checkpoint import save_final_model_ddp
from dataset import (
    SS3_ID2LABEL,
    SS3_LABEL2ID,
    SS8_ID2LABEL,
    SS8_LABEL2ID,
    compute_accuracy,
    create_dataloader,
    get_parameter_names_with_lora,
)
from distributed_config import DistributedConfig
from perf_logger import PerfLogger
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float:
    """Training loop for LoRA fine-tuning of ESM-2 with Transformer Engine and PEFT.

    Args:
        args: Configuration arguments from hydra.

    Returns:
        Final loss value.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    train_dataloader, val_dataloader, train_dataset_or_sampler = create_dataloader(
        distributed_config=dist_config,
        use_sequence_packing=args.use_sequence_packing,
        **OmegaConf.to_container(args.dataset, resolve=True),
    )

    perform_validation = val_dataloader is not None

    # Create a device mesh for DDP. While this isn't strictly necessary, it mirrors the device mesh we create for FSDP2
    # and MFSDP.
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("ddp",))

    # For testing, we don't want to depend on loading pre-trained weights.
    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True, dtype=torch.bfloat16)
    if args.use_sequence_packing:
        config.attn_input_format = "thd"

    if args.dataset["ss3_classification"]:
        config.id2label = SS3_ID2LABEL
        config.label2id = SS3_LABEL2ID
    else:
        config.id2label = SS8_ID2LABEL
        config.label2id = SS8_LABEL2ID

    if args.use_pretrained:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_tag, config=config, trust_remote_code=True, dtype="bfloat16"
        )
    else:
        model = AutoModelForTokenClassification.from_config(config, trust_remote_code=True)

    peft_config = peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS,
        inference_mode=False,
        r=args.lora.r,
        lora_alpha=args.lora.alpha,
        target_modules=list(args.lora.target_modules),
        bias="none",
    )

    peft_model = peft.get_peft_model(model, peft_config)
    peft_model.to(device=device)

    print("----- PEFT Model --------")
    peft_model.print_trainable_parameters()

    # Create optimizer.
    decay_parameters = get_parameter_names_with_lora(peft_model)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in peft_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": args.adamw_kwargs.weight_decay,
        },
        {
            "params": [p for n, p in peft_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **args.adamw_kwargs)

    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    peft_model = torch.nn.parallel.DistributedDataParallel(
        peft_model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["ddp"],
        find_unused_parameters=True,
    )

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        peft_model = torch.compile(peft_model)

    perf_logger = PerfLogger(dist_config, args)

    # Training loop.
    step = 0
    epoch = 0
    while step < args.num_train_steps:
        for batch in train_dataloader:
            perf_logger.log_train_start_time()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa PLW2901

            output = peft_model(**batch)
            loss = output.loss
            loss.backward()

            # Compute and clip gradient norms.
            total_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0).item()

            # Step optimizer.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1

            perf_logger.log_train_end_time()
            # Validation
            avg_val_loss = None
            avg_val_acc = None
            if perform_validation and step % args.validation_interval == 0:
                peft_model.eval()
                val_loss_total = 0.0
                val_correct_total = 0
                val_tokens_total = 0
                val_steps = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = {  # noqa: PLW2901
                            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()
                        }
                        val_output = peft_model(**val_batch)

                        # Loss
                        val_loss_total += val_output.loss.item()

                        # Accuracy
                        logits = val_output.logits
                        labels = val_batch["labels"]
                        correct, total = compute_accuracy(logits, labels)
                        val_correct_total += correct
                        val_tokens_total += total

                        val_steps += 1

                avg_val_loss = val_loss_total / val_steps
                avg_val_acc = val_correct_total / val_tokens_total if val_tokens_total > 0 else 0.0
                print(f"\nStep: {step}: Validation Loss = {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}\n")
                peft_model.train()

            perf_logger.log_step(
                step=step,
                batch=batch,
                outputs=output,
                grad_norm=total_norm,
                lr=optimizer.param_groups[0]["lr"],
                val_loss=avg_val_loss,
                val_acc=avg_val_acc,
            )

            if step >= args.num_train_steps:
                break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        train_dataset_or_sampler.set_epoch(epoch)

    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp" if args.checkpoint.ckpt_dir else None

    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_ddp(
            model=peft_model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
