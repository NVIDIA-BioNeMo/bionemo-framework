---
name: setup-fsdp-training
description: >
  Set up FSDP2 or mFSDP distributed training for a TransformerEngine model.
  Triggers when user asks about distributed training, FSDP, data parallel,
  multi-GPU training, or scaling training.
allowed-tools: Read, Grep, Glob, Write, Edit, Bash, Agent
argument-hint: '[fsdp2|mfsdp]'
---

# Set Up FSDP2 Distributed Training

You are setting up distributed training with PyTorch FSDP2 for a TransformerEngine model. Read the reference files first.

## Reference Files

- `reference/train_fsdp2.py` — Complete FSDP2 training script
- `reference/hydra_defaults.yaml` — Hydra configuration template

## Steps

### 1. Initialize Distributed

```python
import torch
from torch.distributed.device_mesh import init_device_mesh

torch.distributed.init_process_group(backend="nccl")
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)

device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
```

### 2. Create Model on Meta Device

```python
with torch.device("meta"):
    model = MyTEModel(config, fp8_recipe=fp8_recipe)
```

### 3. Apply FSDP Wrapping

```python
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

# FP32 master weights with BF16 compute
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=torch.bfloat16,
)

# Shard individual layers first, then the full model
for layer in model.layers:
    fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)
```

### 4. Initialize Weights After Sharding

```python
model.init_empty_weights()  # Moves from meta to cuda
```

### 5. Create Optimizer (AFTER FSDP wrapping)

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=4e-4, betas=(0.9, 0.98), weight_decay=0.01
)
```

### 6. Training Loop

```python
for step, batch in enumerate(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 7. Distributed Checkpointing

```python
import torch.distributed.checkpoint as dcp

dcp.save({"model": model, "optimizer": optimizer}, checkpoint_id=ckpt_path)
dcp.load({"model": model, "optimizer": optimizer}, checkpoint_id=ckpt_path)
```
