import logging
import os
from typing import NamedTuple
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal
from megatron.core.process_groups_config import ModelCommProcessGroups
import numpy as np
import pytest
import torch
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import HyenaInferenceContext
from nemo.collections.llm.inference import generate
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.io.pl import MegatronCheckpointIO

from bionemo.core.data.load import load
from bionemo.llm.utils.weight_utils import (
    MegatronModelType,
    _key_in_filter,
    _munge_key_megatron_to_nemo2,
    _munge_sharded_tensor_key_megatron_to_nemo2,
)
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.torch import check_fp8_support
from copy import deepcopy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


try:
    from modeling_esm import TEEsmForMaskedLM, EsmTELayer

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron_fsdp import fully_shard

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Activate PyTorch memory profiler.
TORCH_MEMORY_PROFILER = False
if TORCH_MEMORY_PROFILER:
    torch.cuda.memory._record_memory_history(max_entries=250000)

USE_HYBRID_FSDP = True
OUTER_SHARD_STRAT = "optim"
USE_TE = True
BATCH_SIZE = 4
SHARD_STRAT = "optim_grads_params"
FP32_WEIGHTS = True
FP32_GRAD = False
USE_CONTEXT_PARALLEL = True
LEARNING_RATE = 1e-4
SEQ_LENGTH = 1024
NUM_TRAIN_STEPS = 25
# model_name = "esm2_t6_8M_UR50D"
model_name = "esm2_t36_3B_UR50D"
# model_name = "esm2_t48_15B_UR50D"

# Create DP and CP sub-meshes. Use the "leftover" ranks for CP.
# One one node, DP = 4 and CP = 2. Scaling on nodes will scale context.
if USE_CONTEXT_PARALLEL:
    DP_SHARD = 1
    DP_INTER = 1
    CP = 2
    TP = 1
else:
    DP_SHARD = 2
    DP_INTER = 1
    CP = 1
    TP = 1

class DistributedConfig(NamedTuple):
    rank: int
    local_rank: int
    world_size: int

    def is_main_process(self) -> bool:
        return self.rank == 0


def load_weights_sharded_inplace_nemo2_to_mcore(
    model: MegatronModelType,
    distributed_checkpoint_dir: str | Path,
    skip_keys_with_these_prefixes: set[str],
    ckpt_format: Literal["zarr", "torch_dist"] = "torch_dist",
):
    logger.info("Start setting up state dict")
    sharded_state_dict = {
        _munge_key_megatron_to_nemo2(k): _munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not _key_in_filter(
            k, skip_keys_with_these_prefixes
        )  # and "_extra_state" not in k  # extra state is needed for fp8 sharded states
    }
    # Load the checkpoint with strict=false to allow for missing keys (backward compatibility)
    # Error: megatron.core.dist_checkpointing.core.CheckpointingException:
    # Object shard ... module.decoder.final_norm._extra_state/shard_0_1.pt not found
    MegatronCheckpointIO(save_ckpt_format=ckpt_format).load_checkpoint(
        distributed_checkpoint_dir, sharded_state_dict=sharded_state_dict, strict=False
    )

def setup_distributed() -> DistributedConfig:
    dist.init_process_group(backend="nccl")
    dist_config = DistributedConfig(
        rank=dist.get_rank(),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=dist.get_world_size(),
    )
    torch.cuda.set_device(dist_config.local_rank)

    logger.info("Initialized distributed training: %s", dist_config)
    return dist_config


def get_batch_on_this_cp_rank(batch, seq_dim: int = 1, mask_seq_dim: int = 2, cp_group: torch.distributed.ProcessGroup = None):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """
    if cp_group is None:
        # Create a world process group for CP.
        cp_group = torch.distributed.new_group()

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = torch.distributed.get_world_size(group=cp_group)
    if cp_size > 1:
        cp_rank = torch.distributed.get_rank(group=cp_group)
        for key, val in batch.items():
            if val is not None:
                seq_dim = seq_dim if key != 'attention_mask' else mask_seq_dim
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor(
                    [cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu', pin_memory=True
                )
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val

    return batch


dist_config = setup_distributed()

device_mesh = init_device_mesh(
    "cuda", (DP_INTER, DP_SHARD, CP, TP, 1), mesh_dim_names=("dp_inter", "dp_shard", "cp", "tp", "pp")
)
device_mesh[("dp_shard", "cp")]._flatten("dp_cp_shard")
device_mesh[("dp_inter", "dp_shard")]._flatten("dp")
hybrid_fsdp_group = device_mesh[("dp_inter", "dp_shard", "cp")]._flatten("hsdp_group").get_group()

# TODO model stuff here
model_config = llm.Hyena1bConfig(
    use_te=True,
    seq_length=8192,
    vortex_style_fp8=False,
)
ckpt_name = "evo2/1b-8k-bf16:1.0"
ckpt_weights: Path = load(ckpt_name) / "weights"
tokenizer = get_nmt_tokenizer(
    "byte-level",
)
# NOTE: the following line breaks because it deeply depends on megatron state for initialization. This path
#  is deeply coupled to the megatron parallel state. We cannot train megatron models directly with this approach unless
#  we initialize all of megatron's global parallel group management state.
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
get_cuda_rng_tracker().add("model-parallel-rng", seed=1234)

model_comm_pgs = ModelCommProcessGroups(
    tp=device_mesh.get_group("tp"),
    pp=device_mesh.get_group("pp"),
    cp=device_mesh.get_group("cp"),
)
raw_megatron_model = model_config.configure_model(tokenizer, model_comm_pgs=model_comm_pgs).eval().cuda()
device = raw_megatron_model.parameters().__next__().device
load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, ckpt_weights, {}, "torch_dist")
model = Float16Module(model_config, raw_megatron_model)

# Move model to GPU if available
device = torch.device(
    f"cuda:{dist_config.local_rank}" if torch.cuda.is_available() else "cpu"
)
#model.to(device, dtype=torch.bfloat16)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, fused=False)

if SHARD_STRAT != "ddp":
    # FSDP Unit Modules
    fsdp_units = [type(m) for m in raw_megatron_model.modules()]

    # Wrap model in Custom FSDP.
    model, optimizer = fully_shard(
        module=model,
        optimizer=optimizer,
        zero_dp_strategy=SHARD_STRAT,
        fsdp_unit_modules=fsdp_units,
        use_hybrid_fsdp=USE_HYBRID_FSDP,
        outer_dp_sharding_strategy=OUTER_SHARD_STRAT,
        device_mesh=device_mesh,
        dp_shard_dim="dp_cp_shard",
        dp_inter_dim="dp_inter",
        tp_dim="tp",
        hybrid_fsdp_group=hybrid_fsdp_group,
        calculate_per_token_loss=False,
        init_model_with_meta_device=True,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=FP32_GRAD,
        preserve_fp32_weights=FP32_WEIGHTS,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        # Auto-sync gradients after post-backward so the user doesn't need to manually wait.
        sync_grads_each_step=True,
        average_in_collective=False,
    )
else:
    # DDP Alternative
    model = DDP(model, device_ids=[dist_config.local_rank])

input_batch = {
    "input_ids": torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, SEQ_LENGTH), device=device, dtype=torch.long),
    "attention_mask": None,
    "position_ids": torch.arange(SEQ_LENGTH, device=device),
    "loss_mask": torch.ones((BATCH_SIZE, SEQ_LENGTH), device=device, dtype=torch.float32),
}

input_batch['labels'] = input_batch["input_ids"].clone()
input_batch['labels'] = input_batch['labels'][:, 1:]

# Training loop
model.train()
if dist_config.rank == 0:  # Only show progress bar on main process
    progress_bar = tqdm(range(NUM_TRAIN_STEPS), desc="Training", disable=False)


for step in range(NUM_TRAIN_STEPS):

    # Time training step.
    start_time = time.perf_counter()

    # Get batch, and move to GPU.
    batch = deepcopy(input_batch)
    if USE_CONTEXT_PARALLEL:
        batch = get_batch_on_this_cp_rank(batch, 1, 1, device_mesh["cp"].get_group())
    batch = {k: v.to(device) for k, v in batch.items()}

    # TODO(@cye): Nullify the attention mask because TE does not support padded un-packed sequences when using CP.
    # THIS WILL ABSOLUTELY DESTROY MODEL ACCURACY. ONLY USE FOR PERFORMANCE BENCHMARKING.
    if USE_CONTEXT_PARALLEL:
        batch.pop("attention_mask")

    # Forward pass with mixed precision
    # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(**batch)
    loss = outputs # not reduced loss
    loss = loss.mean() # reduced loss with mean.
    # Backward pass
    loss.backward()
    # TODO insert hook to capture and save grads
    # Compute gradient norms.
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

    optimizer.step()
    optimizer.zero_grad()

    # Take snapshot.
    if TORCH_MEMORY_PROFILER and (step == 0 or step == NUM_TRAIN_STEPS // 2):
        snapshot = torch.cuda.memory._snapshot()
        from pickle import dump
        with open(f'snapshot_{SHARD_STRAT}_{step}.pickle', 'wb') as f:
            dump(snapshot, f)

    # Compute step time.
    end_time = time.perf_counter()
    step_time = end_time - start_time

    # Log metrics to wandb on main process
    if dist_config.is_main_process():
        logger.info(
            f"Step {step} loss: {loss.item()}, grad_norm: {total_norm}, lr: {optimizer.param_groups[0]['lr']}, steps_per_second: {1 / step_time}, mem: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
        )
        # wandb.log(
        #     {
        #         "train/loss": loss.item(),
        #         "train/global_step": step,
        #         "train/learning_rate": optimizer.param_groups[0]["lr"],
        #         "train/grad_norm": total_norm,
        #         "train/epoch": step / 1,
        #         "train/steps_per_second": 1 / step_time,
        #         "train/torch_memory_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
        #     }
        # )

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item(), "steps_per_second": 1 / step_time, "mem_in_gb": torch.cuda.max_memory_allocated() / 1024**3})


torch.distributed.barrier()
dist.destroy_process_group()
