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
from megatron_fsdp.utils import FSDPDistributedIndex
from torch.distributed.device_mesh import DeviceMesh
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal
from megatron.core.process_groups_config import ModelCommProcessGroups, GradCommProcessGroups
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
from megatron.core import parallel_state
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
import einops
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself
HAVE_EINOPS = True
HAVE_DTENSOR = True

try:
    from modeling_esm import TEEsmForMaskedLM, EsmTELayer

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron_fsdp import fully_shard

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--cp", type=int, default=1)
parser.add_argument("--tp", type=int, default=1)
args = parser.parse_args()
CP = args.cp
TP = args.tp
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
else:
    DP_SHARD = 2
    DP_INTER = 1

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
            try:
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
            except Exception as e:
                if torch.distributed.get_rank() == 0:
                    breakpoint()
                torch.distributed.barrier()
                raise e

    return batch


dist_config = setup_distributed()


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP,
    #pipeline_model_parallel_size=1,
    #virtual_pipeline_model_parallel_size=1,
    context_parallel_size=CP,
    #expert_model_parallel_size=1,
)








# device_mesh = init_device_mesh(
#     "cuda", (DP_INTER, DP_SHARD, CP, TP, 1), mesh_dim_names=("dp_inter", "dp_shard", "cp", "tp", "pp")
# )
# device_mesh[("dp_shard", "cp")]._flatten("dp_cp_shard")
# device_mesh[("dp_inter", "dp_shard")]._flatten("dp")
# hybrid_fsdp_group = device_mesh[("dp_inter", "dp_shard", "cp")]._flatten("hsdp_group").get_group()

# TODO model stuff here
model_config = llm.Hyena1bConfig(
    use_te=True,
    seq_length=SEQ_LENGTH,
    vortex_style_fp8=False,
    context_parallel_size=CP,
    tensor_model_parallel_size=TP,
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

model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups()
grad_comm_pgs = GradCommProcessGroups()
grad_comm_pgs.dp=parallel_state.get_data_parallel_group()

# _DATA_PARALLEL_GROUP_WITH_CP
grad_comm_pgs.dp_cp=parallel_state.get_data_parallel_group(with_context_parallel=True)

# _EXPERT_DATA_PARALLEL_GROUP
#expt_dp=parallel_state.get_expert_data_parallel_group(),

# _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
grad_comm_pgs.intra_dp_cp=parallel_state.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)

# _INTRA_EXPERT_DATA_PARALLEL_GROUP
#intra_expt_dp=parallel_state.get_expert_data_parallel_group(),

# _INTER_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
#inter_dist_opt=parallel_state.get_inter_distributed_optimizer_instance_group(),

# BEGIN HELPER FUNCTIONS
def _init_dist_index(grad_comm_pgs, model_comm_pgs):
    """
    Initialize the distributed index for the module.
    """
    if not HAVE_DTENSOR:
        raise ImportError(
            "This module requires PyTorch with DTensor support. "
            "Please install a compatible version of PyTorch."
        )

    enable_hsdp = False #self.ddp_config.num_distributed_optimizer_instances > 1
    
    if grad_comm_pgs is None and model_comm_pgs is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        if enable_hsdp:
            dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            inter_fsdp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
            hybrid_fsdp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
        else:
            dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            inter_fsdp_group = None
            hybrid_fsdp_group = None
    elif grad_comm_pgs is not None and model_comm_pgs is not None:
        tp_group = getattr(model_comm_pgs, 'tp', None)
        if enable_hsdp:
            dp_cp_group = grad_comm_pgs.intra_dp_cp
            inter_fsdp_group = grad_comm_pgs.inter_dist_opt
            hybrid_fsdp_group = grad_comm_pgs.dp_cp
        else:
            dp_cp_group = grad_comm_pgs.dp_cp
            inter_fsdp_group = None
            hybrid_fsdp_group = None
    else:
        raise ValueError(
            "Both grad_comm_pgs and model_comm_pgs must be either None or provided together."
        )

    if tp_group is None:
        single_rank_group = dist.new_group(ranks=[dist.get_rank()])
        tp_group = single_rank_group

    if enable_hsdp:
        mesh = _get_hsdp_tp_mesh(inter_fsdp_group, dp_cp_group, tp_group)
        dist_index = FSDPDistributedIndex(
            use_hybrid_fsdp=True,
            hsdp_outer_dp_shard= False, #self.ddp_config.outer_dp_sharding_strategy != "no_shard",
            device_mesh=DeviceMesh.from_group(
                [inter_fsdp_group, dp_cp_group, tp_group],
                device_type="cuda",
                mesh=mesh.tolist(),
                mesh_dim_names=["inter_fsdp_dp", "dp_cp", "tp"],
            ),
            dp_inter_dim="inter_fsdp_dp",
            dp_shard_dim="dp_cp",
            tp_dim="tp",
            hybrid_fsdp_group=hybrid_fsdp_group,
        )
    else:
        mesh = _get_dp_tp_mesh(dp_cp_group, tp_group)
        dist_index = FSDPDistributedIndex(
            device_mesh=DeviceMesh.from_group(
                [dp_cp_group, tp_group],
                device_type="cuda",
                mesh=mesh.tolist(),
                mesh_dim_names=["dp_cp", "tp"],
            ),
            dp_shard_dim="dp_cp",
            tp_dim="tp",
        )

    return dist_index
def _get_hsdp_tp_mesh(inter_fsdp_dp_group, dp_cp_group, tp_group):
    assert HAVE_EINOPS, "einops is not installed. Please install it with `pip install einops`."
    world_size = dist.get_world_size()

    mesh = einops.rearrange(
        torch.arange(world_size),
        "(inter_fsdp_dp fsdp tp) -> inter_fsdp_dp fsdp tp",
        inter_fsdp_dp=inter_fsdp_dp_group.size(),
        tp=tp_group.size(),
    )

    mesh_fsdp_ranks = einops.rearrange(
        mesh,
        'inter_fsdp_dp fsdp tp -> (inter_fsdp_dp tp) fsdp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    fsdp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_fsdp_ranks, fsdp_group_ranks), (
        f"[Megatron-FSDP] FSDP ranks in the mesh {mesh_fsdp_ranks} "
        f"do not match the ranks in the FSDP group {fsdp_group_ranks}."
    )

    mesh_tp_ranks = einops.rearrange(
        mesh,
        'inter_fsdp_dp fsdp tp -> (inter_fsdp_dp fsdp) tp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_tp_ranks, tp_group_ranks), (
        f"[Megatron-FSDP] Tensor Parallel ranks in the mesh {mesh_tp_ranks} "
        f"do not match the ranks in the TP group {tp_group_ranks}."
    )

    mesh_inter_fsdp_dp_ranks = einops.rearrange(
        mesh,
        'inter_fsdp_dp fsdp tp -> (fsdp tp) inter_fsdp_dp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    inter_fsdp_dp_group_ranks = dist.get_process_group_ranks(inter_fsdp_dp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(
        mesh_inter_fsdp_dp_ranks, inter_fsdp_dp_group_ranks
    ), (
        f"[Megatron-FSDP] Inter FSDP Data Parallel ranks in the mesh {mesh_inter_fsdp_dp_ranks} "
        f"do not match the ranks in the Inter FSDP DP group {inter_fsdp_dp_group_ranks}."
    )

    return mesh


def _get_dp_tp_mesh(dp_cp_group, tp_group):
    assert HAVE_EINOPS, "einops is not installed. Please install it with `pip install einops`."
    world_size = dist.get_world_size()

    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    # TODO: Supports configurable (dp, cp, tp) order.
    mesh = einops.rearrange(torch.arange(world_size), "(dp_cp tp) -> dp_cp tp", tp=tp_size)

    mesh_dp_ranks = einops.rearrange(mesh, 'dp_cp tp -> tp dp_cp', tp=tp_size)
    dp_cp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_dp_ranks, dp_cp_group_ranks), (
        f"[Megatron-FSDP] Data Parallel ranks in the mesh {mesh_dp_ranks} "
        f"do not match the ranks in the DP group {dp_cp_group_ranks}."
    )

    mesh_tp_ranks = einops.rearrange(mesh, 'dp_cp tp -> (dp_cp) tp', tp=tp_size)
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    assert _check_mesh_ranks_and_group_ranks_are_consistent(mesh_tp_ranks, tp_group_ranks), (
        f"[Megatron-FSDP] Tensor Parallel ranks in the mesh {mesh_tp_ranks} "
        f"do not match the ranks in the TP group {tp_group_ranks}."
    )

    return mesh


def _check_mesh_ranks_and_group_ranks_are_consistent(mesh_ranks, group_ranks):
    current_rank = dist.get_rank()
    current_ranks = list(filter(lambda ranks: current_rank in ranks, mesh_ranks.tolist()))
    assert len(current_ranks) == 1, (
        f"[Megatron-FSDP] Current rank {current_rank} is not unique in "
        f"the mesh ranks {mesh_ranks.tolist()}."
    )
    assert sorted(current_ranks[0]) == sorted(group_ranks), (
        f"[Megatron-FSDP] Current rank {current_rank} in the mesh ranks "
        f"{mesh_ranks.tolist()} does not match the group ranks {group_ranks}."
    )
    return sorted(current_ranks[0]) == sorted(group_ranks)


# END HELPER FUNCTIONS

fsdp_dist_index = _init_dist_index(grad_comm_pgs, model_comm_pgs)





raw_megatron_model = model_config.configure_model(tokenizer).eval().cuda()
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
        dist_index=fsdp_dist_index,
    )
else:
    # DDP Alternative
    model = DDP(model, device_ids=[dist_config.local_rank])
generator = torch.Generator()
generator.manual_seed(1234)
input_batch = {
    "input_ids": torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.long, generator=generator),
    "attention_mask": None,
    "position_ids": torch.arange(SEQ_LENGTH).expand(BATCH_SIZE, SEQ_LENGTH),
    "loss_mask": torch.ones((BATCH_SIZE, SEQ_LENGTH),  dtype=torch.float32),
}

input_batch['labels'] = torch.roll(input_batch["input_ids"], shifts=-1, dims=0)

# Training loop
model.train()
if dist_config.rank == 0:  # Only show progress bar on main process
    progress_bar = tqdm(range(NUM_TRAIN_STEPS), desc="Training", disable=False)


assert parallel_state.is_pipeline_first_stage()
assert parallel_state.is_pipeline_last_stage()

for step in range(NUM_TRAIN_STEPS):

    # Time training step.
    start_time = time.perf_counter()

    # Get batch, and move to GPU.
    batch = deepcopy(input_batch)
    if USE_CONTEXT_PARALLEL:
        batch = get_batch_on_this_cp_rank(batch, 1, 2, parallel_state.get_context_parallel_group())
        assert batch["labels"].shape[-1] == SEQ_LENGTH // CP
    def send_to_device(v, device):
        if v is None:
            return v
        return v.to(device)
    batch = {k: send_to_device(v, device) for k, v in batch.items()}

    # TODO(@cye): Nullify the attention mask because TE does not support padded un-packed sequences when using CP.
    # THIS WILL ABSOLUTELY DESTROY MODEL ACCURACY. ONLY USE FOR PERFORMANCE BENCHMARKING.
    # if USE_CONTEXT_PARALLEL:
    #     batch.pop("attention_mask")

    # Forward pass with mixed precision
    # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(**batch)
    loss = outputs # not reduced loss
    loss = loss.mean() # reduced loss with mean.
    # Backward pass
    loss.backward()
    # TODO insert hook to capture and save grads
    # Compute gradient norms.
    all_grads = {n:p.grad for n,p in model.named_parameters()}
    torch.save(all_grads, f"cp_{CP}_grads_rnk_{parallel_state.get_context_parallel_rank()}_{step}.pt")
    break
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
