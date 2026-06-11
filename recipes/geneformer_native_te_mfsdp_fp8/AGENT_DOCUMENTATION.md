# AGENT_DOCUMENTATION.md - AI Agent Documentation

## Repository Overview

**Purpose**: Geneformer pretraining with mFSDP (NVIDIA Fully Sharded Data Parallel) and custom PyTorch training loops for single-cell genomics transformer models.

**Domain**: Computational Biology / Single-Cell Genomics / Deep Learning

**Model Type**: BERT-based transformer for masked language modeling on gene expression data

**Complete Source Code**: For comprehensive code analysis, see `internal/gitingest.txt` which contains the complete source code of the repository in a single text file.

## Current Repository Structure

### Core Files

- `train.py` (10KB, 295 lines) - Main training script with distributed training support
- `modeling_bert_te.py` (32KB, 775 lines) - Custom BERT implementation with Transformer Engine support
- `dataset.py` (2.0KB, 58 lines) - Data loading utilities
- `test_geneformer_model.py` (4.5KB, 122 lines) - Model unit tests
- `test_bert_tokenizer.py` (31KB, 783 lines) - Comprehensive tokenizer tests

### Tokenizer Components

- `tokenizer_auto/` - AutoTokenizer-compatible tokenizer directory
- `tokenizer/` - Original tokenizer files
- `geneformer_tokenizer/` - Additional tokenizer configurations
- `token_dictionary.pkl` (770KB) - Gene symbol to token ID mapping
- `vocab.txt` (397KB, 25,427 lines) - Vocabulary file

### Configuration System

- `hydra_config/` - Hydra configuration files
  - `sanity.yaml` - Small test configuration
  - `10m.yaml` - 10M parameter model
  - `10m_te.yaml` - 10M parameter model with Transformer Engine
  - `4b_te.yaml` - 4B parameter model with TE
  - `4b_te_fp8.yaml` - 4B parameter model with FP8 precision

### Data Files

- `genecorpus_500_samples.parquet` (1.8MB) - Sample training data
- `genecorpus_500_samples_lengths.pkl` (1.5KB) - Sequence length metadata
- `data/` - Additional data directory

### Container & Deployment

- `Dockerfile` (706B, 25 lines) - Container build configuration
- `requirements.txt` (216B, 11 lines) - Python dependencies
- `bionemo-geneformer-recipe-2.sqsh` (21GB) - Singularity container
- `bnmo-recipe-geneformer-1.sqsh` (21GB) - Additional container
- `slurm_interactive.sh` (1.6KB, 46 lines) - SLURM job submission script

## Key Components

### 1. Main Training Script (`train.py`)

- **Purpose**: Distributed training with mFSDP
- **Features**:
  - Hydra configuration management
  - Multi-GPU support with torchrun
  - Transformer Engine integration
  - Weights & Biases logging
  - Custom FSDP wrapping

### 2. Model Architecture (`modeling_bert_te.py`)

- **Description**: Custom BERT with Transformer Engine optimization
- **Key Classes**:
  - `BertForMaskedLM` - Main model for masked language modeling
  - `TEBertLayer` - Transformer Engine optimized layer
  - `BertLayer` - Standard BERT layer
  - `BertEncoder` - Encoder with switchable TE/standard layers
- **Features**:
  - Conditional TE usage via `use_te_layers` config
  - FP8 precision support (H100+ GPUs)
  - mFSDP compatibility

### 3. Tokenizer System

- **AutoTokenizer Compatible**: `tokenizer_auto/` directory works with `AutoTokenizer.from_pretrained()`
- **Vocabulary**: 25,426 unique gene tokens
- **Special Tokens**: pad_token_id=0, mask_token_id=1
- **Testing**: Comprehensive test suite in `test_bert_tokenizer.py`

### 4. Data Processing (`dataset.py`)

- **Format**: Parquet files with gene expression sequences
- **Tokenization**: Gene symbols → token IDs via `token_dictionary.pkl`
- **Sequence Length**: Up to 2048 tokens per cell

## Configuration Schema

### Model Configuration

```yaml
model:
  attention_probs_dropout_prob: 0.02    # Attention dropout rate
  hidden_act: relu                      # Activation function
  hidden_dropout_prob: 0.02             # Hidden layer dropout
  hidden_size: 2560                     # Model dimension
  initializer_range: 0.02               # Weight init std
  intermediate_size: 10240              # FFN dimension
  layer_norm_eps: 1.0e-12              # Layer norm epsilon
  max_position_embeddings: 2048         # Max sequence length
  micro_batch_size: 10                  # Per-device batch size
  model_type: bert                      # Architecture type
  num_attention_heads: 40               # Attention heads
  num_hidden_layers: 36                 # Transformer layers
  pad_token_id: 0                       # Padding token ID
  seq_length: 2048                      # Training sequence length
  use_te_layers: true                   # Use Transformer Engine
  vocab_size: 25426                     # Vocabulary size
```

### Training Configuration

```yaml
training:
  learning_rate: 1e-4                   # Optimizer learning rate
  num_train_steps: 1000                 # Total training steps
  num_workers: 4                        # DataLoader workers
  mlm_probability: 0.15                 # Mask probability
  use_fp8: true                         # Enable FP8 precision
```

### WandB Configuration

```yaml
wandb_init_args:
    name: "geneformer-4b-te"            # Experiment name
    project: "bionemo-recipes"          # Project name
    mode: "offline"                     # Run data management
```

### Data Configuration

```yaml
data:
  path: "/workspace/data/Genecorpus-30M/genecorpus_1M_samples.parquet"
```

## Usage Instructions

### Basic Training

```bash
torchrun --nproc_per_node=1 train.py --config-name <config_name>
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=<num_gpus> train.py --config-name <config_name>
```

### Sanity Test

```bash
torchrun --nproc_per_node=1 train.py  # Uses default sanity config
```

### Available Configurations

- `sanity` - Small test configuration
- `10m` - 10M parameter model
- `10m_te` - 10M parameter model with TE
- `4b_te` - 4B parameter model with TE
- `4b_te_fp8` - 4B parameter model with FP8

## Technical Requirements

### Hardware

- **GPU**: NVIDIA GPUs with CUDA support
- **FP8**: Requires compute capability ≥ 8.9 (H100+)
- **Memory**: Varies by model size

### Software Dependencies

- **Base**: `nvcr.io/nvidia/pytorch:25.03-py3`
- **Key Libraries**:
  - `transformers` - HuggingFace transformers
  - `megatron-fsdp==0.1.0rc1` - Megatron-FSDP
  - `transformer_engine` - NVIDIA TE
  - `hydra-core` - Configuration
  - `wandb` - Experiment tracking
  - `datasets` - Data loading

## Model Variants

### 10M Parameter Model

- `hidden_size: 256`
- `num_hidden_layers: 6`
- `num_attention_heads: 4`

### 4B Parameter Model

- `hidden_size: 2560`
- `num_hidden_layers: 36`
- `num_attention_heads: 40`

## Container Usage

### Build Container

```bash
docker build -t <imagename> .
```

### Run Container

```bash
export CONTAINER_NAME=bionemo-recipe-geneformer
export DATA_SOURCE=<yourdata>
docker run -it --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  -v $DATA_SOURCE:$DATA_PATH \
  $CONTAINER_NAME /bin/bash
```

### Weights & Biases Integration

```bash
export WANDB_API_KEY=<yourapikey>
```

## Testing

### Model Tests

```bash
python test_geneformer_model.py
```

### Tokenizer Tests

```bash
python test_bert_tokenizer.py
```

## Code Quality

### Formatting

- **Ruff**: Configuration in `.ruff.toml`
- **Cache**: `.ruff_cache/` directory

### Current Issues

- No pre-commit configuration found
- Tests may need updating based on recent changes

## Data Format

### Input Data

- **Format**: Parquet files with gene expression sequences
- **Tokenization**: Gene symbols → integer tokens
- **Vocabulary**: 25,426 unique gene tokens
- **Sequence Length**: Up to 2048 tokens per cell

### Special Tokens

- `pad_token_id: 0` - Padding
- `mask_token_id: 1` - Masking for MLM
- Other special tokens map to pad_token_id

## Complete Source Code Access

### `internal/gitingest.txt`

- **Purpose**: Complete repository source code in single file
- **Size**: 157KB, 3,727 lines
- **Use Case**: Comprehensive code analysis by AI agents
- **Benefits**:
  - Complete context for cross-file analysis
  - Easy searching and processing
  - All implementation details in one location
  - Understanding code relationships and dependencies

## Output Structure

### Training Outputs

- `outputs/` - Training checkpoints and logs
- `wandb/` - Weights & Biases artifacts
- Hydra output directories with timestamps

### Generated Files

- `.pytest_cache/` - Test cache
- `__pycache__/` - Python bytecode cache
- `.ruff_cache/` - Linting cache

## Common Issues

### FP8 Compatibility

- **Issue**: FP8 on incompatible hardware
- **Solution**: Automatic fallback to BF16
- **Requirement**: Compute capability ≥ 8.9

### Memory Issues

- Reduce `micro_batch_size`
- Use gradient checkpointing
- Increase GPU count for sharding

### Tokenizer Issues

- Use `tokenizer_auto/` for AutoTokenizer compatibility
- Original tokenizer files in `tokenizer/`
- Comprehensive tests in `test_bert_tokenizer.py`

## Development Notes

### Recent Changes

- Extensive tokenizer testing and compatibility fixes
- AutoTokenizer integration
- Model architecture refinements
- Container updates

### Missing Components

- Pre-commit configuration
- Some documentation may be outdated
- Test coverage could be expanded

This repository provides a complete system for pretraining Geneformer models with state-of-the-art optimizations for single-cell genomics research.
