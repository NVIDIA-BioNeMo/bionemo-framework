# SAE: Generic Sparse Autoencoder Package

A domain-agnostic implementation of Sparse Autoencoders (SAEs) for interpretability research. This package provides multiple SAE architectures, training utilities, and evaluation metrics with minimal dependencies.

## Features

- **Multiple Architectures**: ReLU-L1 SAE, Top-K SAE
- **Training Utilities**: Flexible trainer with checkpointing, logging, and evaluation
- **Evaluation Metrics**: Reconstruction quality, fidelity, dead latent tracking
- **Domain-Agnostic**: No biology or domain-specific dependencies

## Installation

### As a standalone package

```bash
pip install -e sae/
```

### From git repository

```bash
pip install git+https://github.com/yourusername/biosae.git#subdirectory=sae
```

### For development (with UV workspace)

```bash
# From repository root
uv sync
```

## Quick Start

```python
import torch
from sae.architectures import ReLUSAE, TopKSAE
from sae.training import Trainer, TrainingConfig
from sae.utils import get_device, set_seed

# Set random seed
set_seed(42)

# Create a synthetic dataset (replace with your embeddings)
embeddings = torch.randn(10000, 512)

# Create SAE model
sae = ReLUSAE(
    input_dim=512,
    hidden_dim=512 * 8,  # 8x expansion
    l1_coeff=1e-3,
)

# Configure training
config = TrainingConfig(
    lr=3e-4,
    n_epochs=10,
    batch_size=4096,
    device=get_device(),
)

# Train
trainer = Trainer(sae, config)
trainer.train(embeddings)
```

## Architecture Classes

### ReLU-L1 SAE (`ReLUSAE`)

Standard sparse autoencoder with ReLU activation and L1 penalty:

```python
from sae.architectures import ReLUSAE

sae = ReLUSAE(
    input_dim=512,
    hidden_dim=4096,
    l1_coeff=1e-3,
    normalize_decoder=True,
)
```

**Parameters:**

- `input_dim`: Dimension of input embeddings
- `hidden_dim`: Number of latent features (typically 4-32x input_dim)
- `l1_coeff`: L1 penalty coefficient (controls sparsity)
- `normalize_decoder`: Whether to normalize decoder weights to unit norm

### Top-K SAE (`TopKSAE`)

Sparse autoencoder with Top-K activation (only top K features per input):

```python
from sae.architectures import TopKSAE

sae = TopKSAE(
    input_dim=512,
    hidden_dim=4096,
    top_k=64,
)
```

**Parameters:**

- `input_dim`: Dimension of input embeddings
- `hidden_dim`: Number of latent features
- `top_k`: Number of active features per input

## Training API

### TrainingConfig

```python
from sae.training import TrainingConfig

config = TrainingConfig(
    lr=3e-4,  # Learning rate
    n_epochs=10,  # Number of epochs
    batch_size=4096,  # Batch size
    device="cuda",  # Device ('cuda', 'cpu', 'mps')
    log_interval=100,  # Log every N steps
    checkpoint_dir="./ckpts",  # Checkpoint directory (None = no checkpointing)
    checkpoint_steps=1000,  # Checkpoint every N steps
)
```

### Trainer

```python
from sae.training import Trainer

trainer = Trainer(sae, config)
trainer.train(embeddings)  # embeddings: [N, input_dim]
```

## Evaluation Metrics

### Reconstruction Quality

```python
from sae.eval import compute_reconstruction_metrics

metrics = compute_reconstruction_metrics(sae, embeddings)
print(f"MSE: {metrics.mse:.4f}")
print(f"Variance Explained: {metrics.variance_explained:.2%}")
```

### Dead Latent Tracking

```python
from sae.eval import DeadLatentTracker

tracker = DeadLatentTracker(hidden_dim=4096, device="cuda")

for batch in dataloader:
    codes = sae.encode(batch)
    tracker.update(codes)

stats = tracker.get_stats()
print(f"Dead latents: {stats['dead_pct']:.1f}%")
```

### Fidelity (for language models)

```python
from sae.eval import evaluate_fidelity

# Requires a language model that produces logits
fidelity = evaluate_fidelity(
    sae=sae,
    model=your_language_model,
    embeddings=embeddings,
    target_logits=original_logits,
)
print(f"Fidelity: {fidelity.fidelity_pct:.2%}")
```

## Utilities

```python
from sae.utils import get_device, set_seed, get_file_limit
from sae.utils import (
    sae_weight_memory,
    sae_forward_memory,
    sae_backward_memory,
    sae_total_memory,
)

# Device detection
device = get_device()  # Returns 'cuda', 'mps', or 'cpu'

# Reproducibility
set_seed(42)

# Memory estimation
total_mem = sae_total_memory(input_dim=512, hidden_dim=4096, batch_size=4096)
print(f"Estimated memory: {total_mem / 1e9:.2f} GB")
```

## Design Philosophy

This package is designed to be:

- **Minimal**: Only essential dependencies (torch, numpy, tqdm)
- **Domain-agnostic**: No biology, NLP, or vision-specific code
- **Extensible**: Easy to subclass `SparseAutoencoder` for custom architectures
- **Research-friendly**: Simple, readable code optimized for experimentation

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@software{sae2024,
  title={SAE: Generic Sparse Autoencoder Package},
  author={Wilber, Jared},
  year={2024},
  url={https://github.com/yourusername/biosae}
}
```
