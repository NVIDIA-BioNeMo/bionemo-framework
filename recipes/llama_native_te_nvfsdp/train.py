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

import logging
import os
import time
from dataclasses import dataclass, field

import hydra
import torch
import torch.distributed as dist
import transformer_engine.pytorch
import wandb
from megatron_fsdp.fully_shard import fully_shard
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM

#from dataset import create_dataloader
from model import NVLlamaForCausalLM, NVLlamaModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DistributedConfig:
    """Class to track distributed ranks."""

    rank: int = field(default_factory=dist.get_rank)
    local_rank: int = field(default_factory=lambda: int(os.environ["LOCAL_RANK"]))
    world_size: int = field(default_factory=dist.get_world_size)

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2_000,
    num_training_steps=500_000,
    last_epoch=-1,
):
    """Linear warmup and decay scheduler for ESM-2 pretraining.

    The description from Lin 2022 is: The learning rate is warmed up over the first 2,000 steps
    to a peak value of 4e-4 (1.6e-4 for the 15B parameter model), and then linearly decayed to
    one tenth of its peak value over the 90% of training duration. We've found internally that a
    longer warmup helps convergence for larger models (3B+) with bf16 precision.
    """
    decay_steps = int(num_training_steps * 0.9)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Warmup phase: linearly increase learning rate
            return float(current_step) / float(max(1, num_warmup_steps))
        # Decay phase: linearly decay to one tenth of peak over 90% of training
        elif current_step > decay_steps:
            return 0.1  # one tenth of peak learning rate after decay period
        else:
            # Linear decay from 1.0 to 0.1 over decay_steps-num_warmup_steps
            return 1.0 - 0.9 * (current_step - num_warmup_steps) / float(max(1, decay_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@hydra.main(config_path="hydra_config", config_name="L0_sanity.yaml", version_base="1.2")
def main(args: DictConfig):
    """Train Llama with TE layers using accelerate.

    Valid model names are:
    - "meta-llama/Meta-Llama-3-8B"
    - "meta-llama/Llama-2-7b-hf"
    """
    # Initialize distributed training and create a device mesh for FSDP.
    # We have to create a dummy mesh dimension for context parallel and tensor parallel for things
    # to work correctly with megatron-fsdp.
    dist.init_process_group(backend="nccl")
    dist_config = DistributedConfig()
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1),
        mesh_dim_names=("fsdp", "tp"),
    )
    device = torch.device(f"cuda:{dist_config.local_rank}")
    logger.info("Initialized distributed training: %s", dist_config)

    if dist_config.is_main_process():
        wandb.init(**args.wandb_init_args, config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True))

    # Create a pretrained LLama model with a masked language model head.
    config = AutoConfig.from_pretrained(
        f"{args.model_name}", trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    # config.max_seq_length = args.max_seq_length
    # config.micro_batch_size = args.micro_batch_size
    model = NVLlamaForCausalLM.from_pretrained(
        f"{args.model_name}",
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Log model and number of parameters on main process.
    if dist_config.is_main_process():
        logger.info("model:\n%s", model)
        logger.info(f"total number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)

    # # Wrap model in megatron-fsdp if we're using it, otherwise wrap with DDP.
    if args.use_fsdp:
        model, optimizer = fully_shard(
            module=model,
            optimizer=optimizer,
            fsdp_unit_modules=[
                transformer_engine.pytorch.TransformerLayer,
                transformer_engine.pytorch.LayerNorm,
                transformer_engine.pytorch.LayerNormLinear,
            ],
            device_mesh=device_mesh,
            dp_shard_dim="fsdp",
            tp_dim="tp",
            **args.fully_shard_kwargs,
        )

    else:
        model = model.to(device=device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank],
            output_device=dist_config.local_rank,
            device_mesh=device_mesh["fsdp"],
        )

    # This is important; the LR scheduler modifies optimizer.step(), so this needs to get created
    # after the optimizer gets wrapped in FSDP. Here we use a warmup and linear decay scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    # Training loop.
    model.train()
    if dist_config.is_main_process():
        progress_bar = tqdm(range(args.num_train_steps), desc="Training", disable=False)

    # Create a dataloader that just infinitely loops over the dataset.
    train_iterator, epoch_len = create_dataloader(
        args.data_path,
        args.micro_batch_size,
        max_length=args.max_seq_length,
    )

    # Training loop.
    previous_step_time = time.perf_counter()
    for step in range(args.num_train_steps):
        # Get batch.
        batch = next(train_iterator)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)

        # Backward pass.
        loss = outputs.loss
        loss.backward()

        # Compute and clip gradient norms.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        # Step optimizer.
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log metrics to logger and wandb on main process.
        if dist_config.is_main_process():
            current_time = time.perf_counter()
            step_time = current_time - previous_step_time
            previous_step_time = current_time

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Step %d loss: %f, grad_norm: %f, lr: %f",
                step,
                loss.detach().item(),
                total_norm,
                current_lr,
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/global_step": step,
                    "train/learning_rate": current_lr,
                    "train/grad_norm": total_norm,
                    "train/epoch": step / epoch_len,
                    "train/step_time": step_time,
                }
            )

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    # Clean up distributed training
    if dist_config.is_main_process():
        wandb.finish()

    dist.destroy_process_group()

    ###################
    ##################
    #################
    # Inference
    model.eval()

    device = next(model.parameters()).device  # get model's device

    # Create a dummy input of zeros
    # Normally you would pass token IDs from the tokenizer, but here you want zeros
    input_ids = torch.zeros((1, 10), dtype=torch.long, device=device)  # batch_size=1, seq_len=10

    # Run inference (no gradients needed)
    with torch.no_grad():
        outputs = model(input_ids)

    # outputs.last_hidden_state is [batch_size, seq_len, hidden_size]
    print(outputs)
    print(outputs.logits[0])
    print(outputs.logits[0].shape)


if __name__ == "__main__":
    main()
