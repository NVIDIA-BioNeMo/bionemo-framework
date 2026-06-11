# `BioNeMo-Vision`: Training a `VisionTransformer` (ViT) with `Megatron-FSDP` and `TransformerEngine`

_Adapted ViT model code from huggingface/pytorch-image-models (TImM) written by Ross Wightman (@rwightman / Copyright 2020), which you can check out here: https://github.com/huggingface/pytorch-image-models_

### Pre-Requisites

#### Docker Container

To build a Docker image for this recipe, run the following commands:

```
docker build -t <image_repo>:<image_tag> .
```

To launch a Docker container from the image, run the following command:

```
# Utilize plenty of shared memory (--shm-size) to support loading large batches of image data!
docker run -it --rm --gpus=all --shm-size=16G <image_repo>:<image_tag>
```

#### PIP Install

If you have a virtual environment and CUDA installed, you can install the recipe's dependencies using `pip`:

```
cd recipes/vit
# If this causes problems, you can add PIP_CONSTRAINT= before the `pip install` command to ignore potentially trivial dependency conflicts.
# We strongly recommend installing into a clean virtual environment or CUDA container, such as the image built from the Dockerfile in this recipe.
pip install -r requirements.txt
```

### Training a Vision Transformer

To train a ViT using FSDP, execute the following command in your Docker container, Python virtual environment, or directly after your `docker run` command:

```
torchrun --nproc-per-node ${NGPU} train.py --config-name vit_base_patch16_224 distributed.dp_shard=${NGPU} training.checkpoint.path=./ckpts/vit
```

This will train on the [`AI-Lab-Makerere/ibean`](https://github.com/AI-Lab-Makerere/ibean/) (HuggingFace: [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans)) dataset and save auto-resumable [Torch DCP](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) checkpoints to the `training.checkpoint.path` directory.

[`train.py`](train.py) is the transparent entrypoint to this script that explains how to modify your own training loop for `Megatron-FSDP` ([PyPI: `megatron-fsdp`](https://pypi.org/project/megatron-fsdp/) / [Source: Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src)) to fully-shard your model across all devices.

The TIMM-derived model code for the ViT can be found in [`vit.py`](vit.py), and data utilities for Beans can be found in [`beans.py`](beans.py).

Various configuration options common in computer vision modeling can be found in [config](./config/).

### Checkpointing

#### Megatron-FSDP DCP

To save Megatron-FSDP distributed checkpoints, refer to the following helper functions in [checkpoint.py](./checkpoint.py):

```python
import torch


def save_dcp_checkpoint(checkpoint_path, model=None, optimizer=None):
    """Save a Torch DCP checkpoint of the model and optimizer to checkpoint_path.

    Docs: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
    """
    # Save model and optimizer checkpoints.
    state_dict = {}
    if model is not None:
        state_dict["model"] = model.state_dict()
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.distributed.checkpoint.save(state_dict, checkpoint_id=checkpoint_path)


def load_dcp_checkpoint(checkpoint_path, model=None, optimizer=None):
    """Load a Torch DCP checkpoint from checkpoint_path into model and optimizer.

    Docs: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
    """
    # Load model and optimizer checkpoints.
    state_dict = {}
    if model is not None:
        state_dict["model"] = model.state_dict()
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.distributed.checkpoint.load(state_dict, checkpoint_id=checkpoint_path)
    if model is not None:
        model.load_state_dict(state_dict["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
```

This can be loaded directly into the `MegatronFSDP` model:

```python
# Create a MegatronFSDP model and optimizer.
model, optimizer = fully_shard(model, optimizer, ...)

# Load Megatron-FSDP DCP checkpoint into model and/or optimizer.
load_dcp_checkpoint(CKPT_PATH, model=model, optimizer=optimizer)
```

#### Checkpoint Conversion

To convert DCP checkpoints to non-distributed Torch checkpoints, and vice-versa, you can run the following command from `torch`:

```
python -m torch.distributed.checkpoint.format_utils --help
usage: format_utils.py [-h] {torch_to_dcp,dcp_to_torch} src dst

positional arguments:
  {torch_to_dcp,dcp_to_torch}
                        Conversion mode
  src                   Path to the source model
  dst                   Path to the destination model

options:
  -h, --help            show this help message and exit
```

For example:

```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch step_75_loss_1.725 torch_ckpt_test.pt
```

or:

```python
from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)

# Convert DCP model checkpoint to torch.save format.
dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_PATH)

# Convert torch.save model checkpoint back to DCP format.
torch_save_to_dcp(TORCH_SAVE_CHECKPOINT_PATH, f"{CHECKPOINT_DIR}_new")
```

#### Megatron-FSDP Checkpoint State Caveats

_Note that `torch.save`-converted distributed checkpoints (DCP) cannot be loaded directly into `MegatronFSDP` module classes, because Megatron-FSDP expects an unevenly-sharded DCP checkpoint with metadata not available in `torch.save` checkpoints that defines the distributed read and write sharding strategy for DCP load and save respectively. To load a non-distributed checkpoint for training with Megatron-FSDP, simply load the checkpoint into the unsharded model before calling `fully_shard` as an alternative to loading in a DCP checkpoint after `fully_shard`!_

```python
from checkpoint import load_torch_checkpoint

# Initialize model.
model = build_vit_model(cfg, device_mesh)

# Load torch.save model checkpoint. If the checkpoint was converted
# from a DCP checkpoint produced by Megatron-FSDP, set megatron_fsdp=True,
# which simply strips the "module." prefix from the state dictionary.
load_torch_checkpoint(CKPT_PATH, model, megatron_fsdp=True)

# Fully-shard.
model, _ = fully_shard(model, ...)
```

TODO(@cspades): For converting DCP directly to HuggingFace SafeTensors checkpoints, you can look into: https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/

### Inference

[infer.py](./infer.py) is an example inference script that loads in a non-distributed `torch.save` checkpoint into an un-sharded ViT.

For inference with Megatron-FSDP, refer to the `fully_shard` + `load_dcp_checkpoint` pattern in [train.py](./train.py) / [checkpoint.py](./checkpoint.py) and described in [Megatron-FSDP DCP](#megatron-fsdp-dcp).
