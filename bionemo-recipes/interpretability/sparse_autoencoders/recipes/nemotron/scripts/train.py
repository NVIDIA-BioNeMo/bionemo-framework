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

r"""Step 2: Train an SAE on cached Nemotron activations.

Reads a sharded activation store produced by ``extract.py`` and trains a TopK
(or ReLU) sparse autoencoder on it, streaming shards from disk so host memory
stays bounded. Extraction and evaluation are separate steps, so you can sweep
many SAE sizes/hparams against a single cached activation store.

This is step 2 of the 3-step Nemotron SAE workflow:
    1. extract.py  -- extract activations from Nemotron-3-Nano
    2. train.py    -- train SAE on cached activations  (this file)
    3. eval.py     -- evaluate SAE (reconstruction + loss recovered)

Usage:
    # Train from a cache produced by extract.py
    python scripts/train.py \
        activations.cache_dir=.cache/activations/nemotron_l39 \
        checkpoint.dir=outputs/k32_8x/checkpoints

    # A scaling experiment (8x / 16x / 32x)
    python scripts/train.py +experiments=topk_k32_8x \
        activations.cache_dir=.cache/activations/nemotron_l39 \
        checkpoint.dir=outputs/k32_8x/checkpoints
"""

import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sae.architectures import ReLUSAE, TopKSAE
from sae.perf_logger import PerfLogger
from sae.training import ParallelConfig, Trainer, TrainingConfig, WandbConfig
from sae.utils import get_device, set_seed


def get_device_from_config(cfg: DictConfig) -> str:
    """Get device from config or auto-detect."""
    if cfg.device is not None:
        return cfg.device
    return get_device()


def build_sae(cfg: DictConfig, input_dim: int) -> torch.nn.Module:
    """Build SAE model from config."""
    hidden_dim = input_dim * cfg.model.expansion_factor

    if cfg.model.type == "topk":
        auxk = cfg.model.get("auxk", None) or None  # treat 0 as disabled
        return TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            top_k=cfg.model.top_k,
            normalize_input=cfg.model.get("normalize_input", False),
            auxk=auxk,
            auxk_coef=cfg.model.get("auxk_coef", 1 / 32),
            dead_tokens_threshold=cfg.model.get("dead_tokens_threshold", 10_000_000),
            decoder_impl=cfg.model.get("decoder_impl", "dense"),
        )
    elif cfg.model.type == "relu":
        return ReLUSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            l1_coeff=cfg.model.get("l1_coeff", 1e-2),
        )
    raise ValueError(f"Unknown model type: {cfg.model.type}")


def build_training_config(cfg: DictConfig, device: str) -> TrainingConfig:
    """Build TrainingConfig from Hydra config."""
    return TrainingConfig(
        lr=cfg.training.lr,
        n_epochs=cfg.training.n_epochs,
        batch_size=cfg.training.batch_size,
        device=device,
        log_interval=cfg.training.log_interval,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        checkpoint_dir=cfg.checkpoint.dir,
        checkpoint_steps=cfg.checkpoint.steps,
        lr_scale_with_latents=cfg.training.get("lr_scale_with_latents", False),
        lr_reference_hidden_dim=cfg.training.get("lr_reference_hidden_dim", 2048),
        max_steps=cfg.training.get("max_steps", None),
    )


def build_wandb_config(cfg: DictConfig) -> WandbConfig:
    """Build WandbConfig from Hydra config."""
    return WandbConfig(
        enabled=cfg.wandb.enabled,
        project=cfg.wandb.project,
        run_name=cfg.wandb.get("run_name"),
        group=cfg.wandb.get("group"),
        job_type=cfg.wandb.get("job_type"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def build_parallel_config(cfg: DictConfig) -> ParallelConfig:
    """Build ParallelConfig from Hydra config."""
    dp_size = 1
    if OmegaConf.select(cfg, "parallel") is not None:
        dp_size = cfg.parallel.get("dp_size", 1)
    return ParallelConfig(dp_size=dp_size)


def build_cached_source(cfg: DictConfig):
    """Open a cached activation store (default path). Returns (input_dim, dataloader, store)."""
    cache_dir = cfg.activations.get("cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "train.py reads activations from a cache. Set activations.cache_dir to a store "
            "produced by extract.py, e.g. activations.cache_dir=.cache/activations/nemotron_l39 "
            "(or enable streaming with streaming.enabled=true)."
        )
    # Hydra changes cwd per run; resolve relative paths against the original cwd.
    cache_path = Path(hydra.utils.get_original_cwd()) / cache_dir
    if not (cache_path / "metadata.json").exists():
        raise FileNotFoundError(
            f"No activation store at {cache_path}. Run extract.py first:\n"
            f"    python scripts/extract.py activations.cache_dir={cache_dir} "
            f"activations.layer={cfg.activations.layer}"
        )

    from sae.activation_store import load_activations

    store = load_activations(cache_path)
    meta = store.metadata
    if meta.get("model_name") != cfg.activations.model_name:
        raise ValueError(f"Cache model mismatch: {meta.get('model_name')} vs {cfg.activations.model_name}")
    if meta.get("layer") != cfg.activations.layer:
        raise ValueError(f"Cache layer mismatch: {meta.get('layer')} vs {cfg.activations.layer}")

    dataloader = store.get_streaming_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.get("shuffle", True),
        seed=cfg.seed,
    )
    print(f"Streaming {meta['n_shards']} shards (~{meta['n_samples']:,} tokens) from {cache_path}")
    return meta["hidden_dim"], dataloader, store


def build_streaming_source(cfg: DictConfig, device: str, streaming_cfg: DictConfig):
    """Producer-consumer: extract activations on the fly (no disk).

    Returns (input_dim, dataloader, producer_factories). Each factory yields fresh
    activation chunks by running a Nemotron replica over a slice of FineWeb;
    sae.streaming runs each in a background thread feeding a shared bounded queue
    to the Trainer.

    Parallelism:
      - streaming.extract_devices empty -> one replica (device_map="auto").
      - streaming.extract_devices=[...]  -> one replica per listed GPU. Under
        torchrun (WORLD_SIZE>1) the GPUs and the text are split across ranks, so
        the SAE consumer trains via DDP (cuda:{rank}) while extraction runs in
        parallel on the listed GPUs. Threads on distinct GPUs run concurrently
        because PyTorch releases the GIL during CUDA execution.
    """
    from nemotron_sae.data import load_fineweb
    from nemotron_sae.models import NemotronModel
    from sae.streaming import StreamingConfig, make_streaming_dataloader

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    texts = load_fineweb(
        split=cfg.data.get("split", "train"),
        max_samples=cfg.data.get("max_samples"),
        min_length=cfg.data.get("min_length", 50),
        subset=cfg.data.get("subset", "sample-10BT"),
    )
    # Rank-disjoint shard so each DDP rank consumes different data.
    if world_size > 1:
        texts = texts[rank::world_size]
    print(f"[rank {rank}] {len(texts)} text samples")

    stream_config = StreamingConfig(
        enabled=True,
        queue_size=streaming_cfg.get("queue_size", 8),
        shuffle_buffer_size=streaming_cfg.get("shuffle_buffer_size", 0),
        seed=cfg.seed,
        drop_last=streaming_cfg.get("drop_last", False),
    )

    extract_devices = list(streaming_cfg.get("extract_devices", []) or [])

    if not extract_devices:
        # Single replica sharded across visible GPUs (device_map="auto").
        print(f"Loading {cfg.activations.model_name} (layer {cfg.activations.layer}) for streaming...")
        nemotron = NemotronModel(
            model_name=cfg.activations.model_name,
            layer=cfg.activations.layer,
            device=device,
            max_length=cfg.activations.max_length,
        )

        def producer_factory():
            return nemotron.stream_activations(texts, batch_size=cfg.activations.batch_size)

        dataloader = make_streaming_dataloader(
            producer_factory, batch_size=cfg.training.batch_size, config=stream_config
        )
        print(f"Streaming from Nemotron (no disk cache); input_dim={nemotron.hidden_size}")
        return nemotron.hidden_size, dataloader, [producer_factory]

    # Multi-GPU: one replica per extractor device, split across ranks.
    per_rank = max(1, len(extract_devices) // world_size)
    my_devices = extract_devices[rank * per_rank : (rank + 1) * per_rank] or [
        extract_devices[rank % len(extract_devices)]
    ]
    print(f"[rank {rank}] extractor GPUs: {my_devices} | SAE trains on {device}")

    replicas = []
    for d in my_devices:
        print(f"[rank {rank}] loading replica on cuda:{d}...")
        replicas.append(
            NemotronModel(
                model_name=cfg.activations.model_name,
                layer=cfg.activations.layer,
                device=f"cuda:{d}",
                max_length=cfg.activations.max_length,
                device_map={"": d},
            )
        )
    input_dim = replicas[0].hidden_size

    # Each replica gets a disjoint slice of this rank's text.
    factories = []
    for i, replica in enumerate(replicas):
        shard = texts[i :: len(replicas)]

        def factory(replica=replica, shard=shard):
            return replica.stream_activations(shard, batch_size=cfg.activations.batch_size)

        factories.append(factory)

    dataloader = make_streaming_dataloader(factories, batch_size=cfg.training.batch_size, config=stream_config)
    print(f"[rank {rank}] streaming from {len(replicas)} replicas (no disk cache); input_dim={input_dim}")
    return input_dim, dataloader, factories


def train_tensor_parallel(cfg: DictConfig) -> float:
    """Tensor-parallel training: shard the SAE latents across GPUs (torchrun).

    Trains a ShardedTopKSAE from a cached activation store (TP needs all GPUs for the
    SAE shards, so extract to cache first). Each rank trains on the same data; only the
    replicated pre_bias grad is all-reduced. Launch with torchrun --nproc_per_node=tp_size.
    """
    import torch.distributed as dist
    from sae.activation_store import load_activations
    from sae.architectures import ShardedTopKSAE
    from sae.parallel import train_tp_loop

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != cfg.parallel.tp_size:
        raise RuntimeError(
            f"tp_size={cfg.parallel.tp_size} but WORLD_SIZE={world_size}; use torchrun --nproc_per_node={cfg.parallel.tp_size}"
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # Cache (required for TP). All ranks read the same data (TP replicates the batch).
    cache_dir = cfg.activations.get("cache_dir", None)
    if not cache_dir:
        raise ValueError(
            "Tensor-parallel training requires a cached store (activations.cache_dir); run extract.py first."
        )
    cache_path = Path(hydra.utils.get_original_cwd()) / cache_dir
    store = load_activations(cache_path)
    meta = store.metadata
    if meta.get("layer") != cfg.activations.layer:
        raise ValueError(f"Cache layer mismatch: {meta.get('layer')} vs {cfg.activations.layer}")
    input_dim = meta["hidden_dim"]
    dataloader = store.get_streaming_dataloader(
        batch_size=cfg.training.batch_size, shuffle=cfg.training.get("shuffle", True), seed=cfg.seed
    )

    hidden_dim = input_dim * cfg.model.expansion_factor
    sae = ShardedTopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=cfg.model.top_k,
        rank=rank,
        world_size=world_size,
        normalize_input=cfg.model.get("normalize_input", False),
        auxk=(cfg.model.get("auxk", 0) or None),
        auxk_coef=cfg.model.get("auxk_coef", 1 / 32),
        dead_tokens_threshold=cfg.model.get("dead_tokens_threshold", 10_000_000),
        decoder_impl=cfg.model.get("decoder_impl", "triton"),
    )
    if rank == 0:
        print(
            f"ShardedTopKSAE: global hidden_dim={hidden_dim:,} ({hidden_dim // world_size:,}/rank), input_dim={input_dim}"
        )

    sae = sae.to(device)
    # Optional pre_bias init from data (geometric median). pre_bias is replicated, but each
    # rank's dataloader sample differs, so init on rank 0 and broadcast to keep it in sync.
    if cfg.model.get("init_pre_bias", False):
        if rank == 0:
            sample = next(iter(dataloader))
            sample = sample[0] if isinstance(sample, (tuple, list)) else sample
            sae.init_pre_bias_from_data(sample)
            print("Initialized pre_bias from data (geometric median)")
        dist.broadcast(sae.pre_bias.data, src=0)

    # PerfLogger + W&B on rank 0 only (same logging path as the dense recipe).
    perf_logger = None
    if rank == 0:
        from sae.perf_logger import PerfLogger

        if cfg.wandb.enabled:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.get("run_name"),
                group=cfg.wandb.get("group"),
                job_type=cfg.wandb.get("job_type"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        perf_logger = PerfLogger(
            log_interval=cfg.training.log_interval, use_wandb=cfg.wandb.enabled, print_logs=True, device=device
        )

    ckpt_dir = cfg.checkpoint.dir
    ckpt_path = str(Path(hydra.utils.get_original_cwd()) / ckpt_dir) if ckpt_dir else None
    final_loss = train_tp_loop(
        sae,
        dataloader,
        lr=cfg.training.lr,
        max_steps=cfg.training.get("max_steps") or 1000,
        device=device,
        log_interval=cfg.training.log_interval,
        max_grad_norm=cfg.training.get("max_grad_norm"),
        checkpoint_dir=ckpt_path,
        perf_logger=perf_logger,
    )
    if dist.is_initialized():
        dist.destroy_process_group()
    return final_loss


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train an SAE on Nemotron activations (cached store, streamed, or tensor-parallel)."""
    print(OmegaConf.to_yaml(cfg))

    if int(OmegaConf.select(cfg, "parallel.tp_size", default=1) or 1) > 1:
        return train_tensor_parallel(cfg)

    set_seed(cfg.seed)
    device = get_device_from_config(cfg)
    print(f"Using device: {device}")

    # --- Choose the activation source: streamed on the fly, or a cached store ---
    streaming_cfg = OmegaConf.select(cfg, "streaming", default=None)
    use_streaming = bool(streaming_cfg and streaming_cfg.get("enabled", False))

    store = None
    producer_factories = None
    if use_streaming:
        input_dim, train_data, producer_factories = build_streaming_source(cfg, device, streaming_cfg)
    else:
        input_dim, train_data, store = build_cached_source(cfg)

    # --- Build SAE ---
    sae = build_sae(cfg, input_dim)
    print(f"SAE: {cfg.model.type}, input_dim={input_dim}, hidden_dim={sae.hidden_dim}")

    # Optional: initialize pre_bias from the geometric median of a data sample.
    if cfg.model.get("init_pre_bias", False) and hasattr(sae, "init_pre_bias_from_data"):
        print("Initializing pre_bias from geometric median of data...")
        if use_streaming:
            sample = next(iter(producer_factories[0]()))  # one fresh producer chunk
        else:
            sample = torch.from_numpy(store._load_shard(0)).float()
        sae.init_pre_bias_from_data(sample[: min(32768, sample.shape[0])])
        print(f"  pre_bias initialized (mean={sae.pre_bias.mean().item():.4f})")

    # --- Train ---
    perf_logger = PerfLogger(
        log_interval=cfg.training.log_interval,
        use_wandb=cfg.wandb.enabled,
        print_logs=True,
        device=device,
    )
    trainer = Trainer(
        sae,
        build_training_config(cfg, device),
        wandb_config=build_wandb_config(cfg),
        perf_logger=perf_logger,
        parallel_config=build_parallel_config(cfg),
    )

    # Under DDP each rank already streams its own disjoint text shard, so tell the
    # Trainer the data is pre-sharded (skip any distributed sampler logic).
    data_sharded = build_parallel_config(cfg).dp_size > 1
    final_loss = trainer.fit(
        train_data, max_grad_norm=cfg.training.get("max_grad_norm", None), data_sharded=data_sharded
    )
    print(f"\nTraining complete. Final loss: {final_loss:.6f}")
    if cfg.checkpoint.dir:
        print(f"Checkpoint: {Path(cfg.checkpoint.dir) / 'checkpoint_final.pt'}")
    print("Run scripts/eval.py to evaluate reconstruction + loss recovered.")
    return final_loss


if __name__ == "__main__":
    main()
