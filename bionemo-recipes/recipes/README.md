# Recipes Directory

This directory contains self-contained training examples that demonstrate best practices for scaling
biological foundation models using [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
and [megatron-FSDP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src)). Each recipe is a complete Docker environment with
benchmarked training scripts that users can learn from and adapt for their own research.

## Philosophy

Recipes are designed as **educational reference implementations** rather than production training frameworks. Our guiding principles:

### Code as Documentation

**Users will read your code far more often than they execute it directly.** Prioritize clarity and educational value:

- Write code that demonstrates training features and techniques
- Include detailed comments explaining design decisions
- Make trade-offs explicit in your implementation
- Minimize branching logic and complex abstractions

### Self-Contained Simplicity

Each recipe is a completely isolated environment:

- **No shared dependencies** between recipes
- **No bionemo-recipes imports** from other parts of bionemo-recipes
- **Everything needed** to run training is included in the recipe directory
- **Pinned dependencies** for reproducible results. Eventually we will use a uv lockfile to make automated package updates easier.

### KISS (Keep It Simple) over DRY (Don't Repeat Yourself)

Prioritize **readability and educational value** over code reuse:

- **Duplicate code between recipes is fine** if it makes the training script more readable
- **Keep it simple** - users should understand the full training loop at a glance
- **One concept per recipe** - don't try to demonstrate every feature in one script. If multiple
  features share most of their common infrastructure, use a single recipe folder and multiple
  `train_{feature}.py` entrypoints.

## Adding a New Recipe

### Recipe Naming Convention

Follow this naming pattern to clearly communicate what your recipe demonstrates:

```
{model_name}_{training_framework}_{key_features}/
```

Examples:

- `esm2_native_te/` - ESM-2 with vanilla PyTorch, TransformerEngine, and megatron-fsdp
- `geneformer_native_te_mfsdp_fp8/` - Geneformer with native PyTorch, TransformerEngine, and mixed FSDP

### Required Directory Structure

Each recipe must follow this example layout:

```
recipes/{recipe_name}/
├── README.md                                 # Recipe-specific documentation
├── Dockerfile                                # Self-contained training environment
├── .dockerignore                             # Docker build optimization
├── .ruff.toml                                # Code formatting configuration
├── requirements.txt                          # Pinned Python dependencies
├── hydra_config/                             # Training configuration management
│   ├── defaults.yaml                         # Shared configuration defaults
│   ├── L0_sanity.yaml                        # Fast CI/CD test config
│   ├── L1_{model_size}_perf.yaml             # Performance benchmark config
│   ├── L1_{model_size}_partial_conv.yaml     # Partial convergence test config (optional)
│   └── L2_{model_size}_full_conv.yaml        # Full convergence test config (optional)
├── train.py                                  # Main training entrypoint
├── train_{feature}.py                        # Alternative training scripts (optional)
├── test_train.py                             # L0 and L1 test suite
├── modeling_{model}.py                       # Model definition (if needed)
├── dataset.py                                # Data loading utilities
├── train.parquet                             # Small test dataset (< 5MB)
└── slurm.sh                                  # Example multi-node SLURM script
```

## Implementation Requirements

### Self-Contained Docker Environment

Your `Dockerfile` should create a complete, reproducible training environment:

```dockerfile
FROM nvcr.io/nvidia/pytorch:26.04-py3

# Install dependencies with caching for faster builds
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/requirements.txt \
    PIP_CONSTRAINT= pip install -r /requirements.txt

# Set workspace to avoid pytest conflicts
WORKDIR /workspace/bionemo
COPY . .

# Default command for interactive development
CMD ["/bin/bash"]
```

**Key requirements:**

- Use the latest NVIDIA PyTorch base image unless specific requirements dictate otherwise
- Pin all dependencies in `requirements.txt` for reproducibility
- Include everything needed to run training without external dependencies
- Optimize for Docker layer caching

### Readable Training Scripts

Your `train.py` should be educational and self-explanatory:

```python
#!/usr/bin/env python3
"""
ESM-2 training with TransformerEngine and megatron-fsdp.

This script demonstrates how to:
1. Load and prepare biological sequence data
2. Initialize ESM-2 with TransformerEngine layers
3. Configure megatron-fsdp for memory-efficient multi-GPU training
4. Implement a training loop with proper checkpointing

Key design decisions:
- We use megatron-fsdp ZeRO-3 for maximum memory efficiency
- TransformerEngine FP8 is enabled for H100+ hardware
- Context parallelism handles long biological sequences
"""

import hydra
from omegaconf import DictConfig
import torch
from torch.distributed import init_process_group, destroy_process_group


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig):
    """Main training entrypoint."""

    # 1. Initialize distributed training
    init_process_group(backend="nccl")

    # 2. Load model and data (with clear explanations)
    model = load_model(args.model_config)
    dataloader = create_dataloader(args.data_config)

    # 3. Configure optimization strategy
    optimizer = setup_optimizer(model, args.optimizer_config)

    # 4. Training loop with checkpointing
    train_loop(model, dataloader, optimizer, args)

    destroy_process_group()


if __name__ == "__main__":
    main()
```

**Key principles:**

- **Comprehensive docstrings** explaining what the recipe demonstrates
- **Inline comments** for non-obvious design decisions
- **Modular functions** with clear responsibilities
- **Error handling** for common failure modes
- **Progress logging** so users understand training status

### Hydra Configuration Management

Use Hydra for clean, hierarchical configuration management:

#### `defaults.yaml` - Shared Configuration

```yaml
# Model configuration
model:
  name: esm2_t12_35M_UR50D
  use_te: true
  fp8_training: false

# Training configuration
training:
  micro_batch_size: 4
  gradient_accumulation_steps: 1
  num_train_steps: 1000
  save_interval: 100

# Optimization configuration
optimizer:
  type: adamw
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.98]

# Distributed training
distributed:
  backend: nccl
  mfsdp:
    enable: true
    sharding_strategy: zero3

# Logging
wandb:
  project: bionemo-recipes
  mode: online
```

#### `L0_sanity.yaml` - Fast Development Testing

```yaml
defaults:
  - defaults

# Override for fast testing
model:
  name: esm2_t6_8M_UR50D  # Smallest model for speed

training:
  micro_batch_size: 2
  num_train_steps: 5      # Minimal steps for CI/CD
  save_interval: 5

wandb:
  mode: offline           # No external logging in CI
```

#### `L1_benchmark.yaml` - Performance Validation

```yaml
defaults:
  - defaults

# Configuration for performance benchmarking
model:
  name: esm2_t12_35M_UR50D
  fp8_training: true      # Enable FP8 for performance

training:
  micro_batch_size: 8
  num_train_steps: 100    # Enough steps for stable metrics

wandb:
  name: "esm2_mfsdp_benchmark"
  tags: ["L1", "benchmark", "performance"]
```

Whenever possible, initialize objects by directly passing config objects as `**kwargs`:

```python
model = MyModel(**config.model_kwargs)
optimizer = AdamW(**config.optimizer_kwargs)
```

### Comprehensive Testing

Ensure the following tests are done when implementing.

#### L0 Tests - Fast CI/CD Validation

Your `test_train.py` must include L0 tests that run quickly in CI/CD. Include both tests that
execute `main` in the same process, as well as tests that call torchrun / accelerate launch as we
expect a user to do.

```python
def test_train_main_in_same_process(monkeypatch, session_temp_dir: Path):
    """Test that train.py runs successfully with sanity config and creates expected outputs."""

    # Get the recipe directory
    recipe_dir = Path(__file__).parent

    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp8")
    monkeypatch.setenv("ACCELERATE_FP8_BACKEND", "TE")

    with initialize_config_dir(
        config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"
    ):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[f"trainer.output_dir={session_temp_dir}"],
        )

    main(sanity_config)


@pytest.mark.parametrize(
    "accelerate_config", ["some_config.yaml", "some_other_config.yaml"]
)
def test_accelerate_launch(accelerate_config, tmp_path):
    """Test that accelerate launch runs successfully."""
    # Run 'accelerate launch train.py' as a subprocess
    subprocess.run(
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(accelerate_config_path),
            "train.py",
            "--config-name",
            "L0_sanity",
            f"trainer.output_dir={tmp_path}",
        ],
        cwd=recipe_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        timeout=240,
    )
```

#### L1 Benchmark Tests

L1 tests should be specified using a `L1_{model_name}_{test_type}.yaml` config file. These should be
workflows that will be launched using a common SLURM or Lepton batch script, complete in under 4
hours, and have a clear set of performance metrics to validate.

### Cluster-Agnostic SLURM Script

Provide a reference SLURM script that works across different cluster configurations:

```bash
#!/bin/bash
#SBATCH --nodes=2                         # number of nodes
#SBATCH --ntasks-per-node=1    	          # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00                   # wall time
#SBATCH --mem=0                 	      # all mem avail

set -x -e
ulimit -c 0

export GPUS_PER_NODE=8
export CMD="accelerate launch \
    --config_file accelerate_config/some_config.yaml \
    --machine_rank "\$SLURM_NODEID" \
    --num_machines "$SLURM_NNODES" \
    --main_process_ip "\$SLURM_SRUN_COMM_HOST" \
    --main_process_port 12340 \
    --num_processes "$(( $SLURM_NNODES * $GPUS_PER_NODE ))" \
    train.py
"

# Mount a persistent cache directory to cache dataset downloads and transformations.
export CACHE_DIR=<cache_dir>

srun \
  --container-image=<image_name> \
  --container-mounts=${PWD}:/workspace/bionemo,$HOME/.netrc:/root/.netrc,$CACHE_DIR:/root/.cache/huggingface \
  bash -c "$CMD"
```

**SLURM Script Requirements:**

- Don't expose any internal cluster-specific hardware details, environment variables, or file paths.
- Demonstrate how to properly format and launch a multi-node SLURM job with the given framework's
  launcher (e.g. `accelerate launch`, `torchrun`, etc.).

## Quality Standards

### Documentation Requirements

Each recipe must include a detailed README.md covering:

- **What it demonstrates**: Clear statement of the training techniques shown
- **Hardware requirements**: Minimum and recommended GPU configurations
- **Performance expectations**: Benchmark results on reference hardware
- **Configuration options**: How to modify the recipe for different use cases
- **Troubleshooting**: Common issues and solutions

### Performance Benchmarking

Document performance metrics for your recipe, for example:

```markdown
## Performance Benchmarks

### Single Node (8x H100)
- **Throughput**: 2,500 tokens/sec
- **Memory Usage**: 45GB per GPU
- **Model**: ESM-2 650M parameters
- **Batch Size**: 32 (micro_batch_size=4, gradient_accumulation=8)

### Multi Node (2x8 H100)
- **Throughput**: 4,800 tokens/sec
- **Scaling Efficiency**: 96%
- **Network**: InfiniBand
```

## CI/CD Integration

Each recipe must pass the standard test contract, which is a simple pytest invocation:

```bash
cd recipes/my_recipe
docker build -t my_recipe .
docker run --rm -it --gpus all my_recipe pytest -v .
```

**L0 tests** run automatically on every PR and must complete in under 10 minutes.
**L1 tests** run nightly on dedicated hardware and can take up to 4 hours.

## Examples

For reference implementations, examine existing recipes:

- **`esm2_native_te/`**: Comprehensive example showing vanilla PyTorch with TE and megatron-fsdp
- **`geneformer_native_te_mfsdp_fp8/`**: Geneformer with native PyTorch, TransformerEngine, and mixed FSDP

## Best Practices

### What Makes a Great Recipe

- **Educational value**: Users learn something new about scaling biological models
- **Production relevance**: Techniques are applicable to real research workflows
- **Performance validation**: Benchmarked results demonstrate clear benefits
- **Adaptation friendly**: Users can easily modify for their specific needs

### Common Pitfalls to Avoid

- **Over-abstraction**: Don't hide important details behind complex abstractions
- **Feature creep**: Resist adding every possible feature to one recipe
- **Brittle dependencies**: Pin versions and test with fresh environments

Remember: A great recipe teaches users how to scale their biological foundation models effectively.
Focus on clarity, education, and practical applicability over comprehensive use case coverage or
error handling.
