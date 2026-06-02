**🤖 AI AGENTS: START HERE → [AGENT_DOCUMENTATION.md](AGENT_DOCUMENTATION.md)**

# ⚠️ IMPORTANT FOR AI AGENTS ⚠️

**DO NOT proceed without reading [AGENT_DOCUMENTATION.md](AGENT_DOCUMENTATION.md) first.**
This file contains comprehensive documentation specifically designed for AI agents. Please see [gitingest.sh](./internal/gitingest.sh) for the complete codebase.

# Geneformer Pretraining with mfsdp and a custom pytorch training loop.

The code runs inside of a container. To construct this container refer to [container build](#container-build) and [container run](#container-run). In this folder, we supply a pretraining script capable of training several variants of [Geneformer](https://huggingface.co/ctheodoris/Geneformer). Those variants are located in our [hydra_config](hydra_config/). This code was forked from the original geneformer repository, and enhanced to increase its performance.

[Geneformer](https://www.nature.com/articles/s41586-023-06139-9) is a BERT-based transformer pretrained on single-cell transcriptomes. For more information, refer to the nature paper [here](https://www.nature.com/articles/s41586-023-06139-9).

## Training Commands

### Basic Training

```bash
# Single GPU training
torchrun --nproc_per_node=1 train.py --config-name <config_name>

# Multi-GPU training (single node)
torchrun --nproc_per_node=<num_gpus> train.py --config-name <config_name>

# Quick sanity check with included dataset
torchrun --nproc_per_node=1 train.py
```

> **Note:** The config name is the filename without `.yaml` extension (for example, `4b` for `4b.yaml`).

### Advanced Configuration

You can override config parameters directly from the command line. The most common options are:

- `model.use_te_layers=True/False` - Enable/disable Transformer Engine layers
- `training.use_fp8=True/False` - Enable/disable FP8 precision (requires H100+ GPUs)
- `training.use_mfsdp=True/False` - Enable/disable mfsdp distributed training

#### Examples

```bash
# Run 106M model with default settings
torchrun --nproc_per_node=1 train.py --config-name 106m

# Run 106M model without Transformer Engine layers
torchrun --nproc_per_node=2 train.py --config-name 106m model.use_te_layers=False

# Run 4B model with FP8 and mfsdp enabled
torchrun --nproc_per_node=4 train.py --config-name 4b training.use_fp8=True training.use_mfsdp=True
```

## Configuration Files

We provide pre-built configuration files for different Geneformer model sizes. Each configuration supports optional features like Transformer Engine and FP8 precision.

### Available Models

- **10M parameters**: `10m.yaml`
- **106M parameters**: `106m.yaml`
- **4B parameters**: `4b.yaml`

### Testing Configuration

```bash
# Test your configuration settings
python train.py --config-name l0_sanity --cfg all
```

> **Important:** FP8 precision requires GPU compute capability ≥ 8.9 (H100+ GPUs). Disable FP8 mode on older hardware.

```yaml
# Model configuration for 4B parameter model
model:  # A group of parameters related to the model
  attention_probs_dropout_prob: 0.02  # Dropout probability applied to attention weights to prevent overfitting
  hidden_act: relu  # Activation function used in the feedforward network (relu, gelu, swish, etc.)
  hidden_dropout_prob: 0.02  # Dropout probability applied to hidden states throughout the model
  hidden_size: 2560  # The main dimensionality of the model's hidden representation. Input/output dimension of the attention layers.
  initializer_range: 0.02  # Standard deviation for weight initialization (controls how weights are randomly initialized)
  intermediate_size: 10240  # The width / expanded dimension used inside the feedforward network (FFN).
  layer_norm_eps: 1.0e-12  # Small epsilon value added to layer normalization for numerical stability
  max_position_embeddings: 2048  # Maximum sequence length the model can handle (positional encoding limit)
  micro_batch_size: 10  # The batch size per each device
  model_type: bert  # The architecture of the transformer
  num_attention_heads: 40  # Number of parallel attention heads in multi-head attention (must divide hidden_size evenly)
  num_hidden_layers: 36  # Number of transformer layers stacked in the model (depth of the network)
  pad_token_id: 0  # Token ID used for padding sequences to the same length
  seq_length: 2048  # Maximum sequence length for training (should be <= max_position_embeddings)
  use_te_layers: true  # Whether or not to use transformer engine layers in the model. If set to false we will use regular vanilla bert.
  vocab_size: 25426  # Size of the vocabulary (number of unique tokens the model can process)

# Training configuration
training:
  learning_rate: 1e-4  # The learning rate for the optimizer
  num_train_steps: 1000  # Total number of training steps to perform.
  num_workers: 4  # Number of worker processes for data loading (parallelizes data preprocessing)
  mlm_probability: 0.15  # Probability of masking tokens for masked language modeling (typically 15%)
  use_fp8: true  # Set to true to enable FP8 training
  wandb_init_args:
    name: "geneformer-4b-te"  # Name of the experiment run for tracking in Weights & Biases
    project: "bionemo-recipes"  # Project name to organize runs in Weights & Biases
  checkpoint_dir: "/workspace/bionemo/checkpoints/sanity_te" # Where you want to save your checkpoints.
  save_every_n_steps: 50 # What interval you want to save checkpoints at.
  resume_from_checkpoint: true  # if you want to resume from a checkpoint. If true, we will load the checkpoint with the highest "step count" from the "checkpoint_dir".

# Data configuration
data:
  path: "/workspace/data/Genecorpus-30M/genecorpus_1M_samples.parquet"  # Path to the training dataset file
```

For detailed model-specific configuration files, refer to the [hydra_config/model](./hydra_config/model) directory. Some example configs have already been provided such as
You can find the full configuration for the 4B parameter model in [`hydra_config/model/4b.yaml`](./hydra_config/model/4b.yaml).

## Checkpoint Management

Training jobs often run for many hours and may need to be stopped and restarted. This implementation provides built-in checkpoint support for seamless training resumption.

**Checkpoint Behavior:**

- **Saving**: Checkpoints are automatically saved every `save_every_n_steps` to the `checkpoint_dir`
- **Resuming**: Set `resume_from_checkpoint: true` to automatically resume from the latest checkpoint (latest == highest step count)
- **Fresh start**: Set `resume_from_checkpoint: false` to start training from step 0

When resuming, training will start at the step count where the most recent checkpoint was saved and continue until `num_train_steps` is reached. If no valid checkpoint is found, training starts from step 0.

Checkpoint resuming is supported for both **mfsdp** (distributed checkpoints) and **DDP** (single-file checkpoints) configurations.

### Safetensors Export

At the end of training, the model is automatically exported in safetensors format to the `final_model` directory within your checkpoint directory. This export works for both mfsdp and vanilla DDP training configurations.

**Export Location:**

```
{checkpoint_dir}/final_model/
├── model.safetensors      # Model weights in safetensors format
├── config.json           # Model configuration
```

**How it works:**

- For **mfsdp**: Parameters are gathered from all processes, then saved by rank 0
- For **DDP**: The underlying model is unwrapped and saved by rank 0
- Export happens automatically after training completes successfully

#### Loading Exported Models

You can load the exported model using `BertForMaskedLM.from_pretrained()` for inference or further fine-tuning:

```python
from modeling_bert_te import BertForMaskedLM
import torch

# Load the trained model
model_path = "/workspace/bionemo/checkpoints/your_run/final_model"
model = BertForMaskedLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)

# Example 1: Model inference
model.eval()
with torch.no_grad():
    # Your input tokens here
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Replace with actual token IDs
    outputs = model(input_ids)
    predictions = outputs.prediction_logits

# Example 2: Continue fine-tuning
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Your fine-tuning loop here
for batch in your_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Benefits of Safetensors:**

- **Fast loading**: Faster than pickle-based formats
- **Safe**: No arbitrary code execution risks
- **Memory efficient**: Zero-copy loading when possible
- **Cross-platform**: Works across different PyTorch versions

## Containers

### Container Build

This folder contains its own [Dockerfile](Dockerfile) and [requirements](requirements.txt) used for creating your workload environment.
If you want to create your own container, you should run the following BUILD command:

```bash
docker build -t <imagename> .
```

where `.` is expected to be a folder containing the `Dockerfile`.

### Dependencies

Our main dependency is the pytorch container specified inside the [Dockerfile](Dockerfile). Other than that we have pip packages listed inside [requirements.txt](requirements.txt) for python specific packages.

### Container Run

Configure dataset paths in your hydra config's `data.path` variable using absolute paths.

Example run command:

```bash
export CONTAINER_NAME=bionemo-recipe-geneformer
export DATA_SOURCE=/path/to/your/data
export DATA_PATH=/workspace/data

docker run -it --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  -v $DATA_SOURCE:$DATA_PATH \
  $CONTAINER_NAME /bin/bash
```

### WandB

We support full integration with weights and biases. To use this, the environment variable:nter

```
export WANDB_API_KEY=<yourapikey>
```

Also, enter your experiment name and project in the hydra config section `wandb_init_args`.

### Dataset

This repository has two files associated with our dataset. There is a parquet file with 500 samples of tokenized data that originated from the HF [Geneformer](https://huggingface.co/ctheodoris/Geneformer/tree/main). Additionally, there is a `vocab.txt` file that holds the full vocabulary.
