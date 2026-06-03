# bionemo-noodles

`bionemo-noodles` is a Python wrapper of [noodles](https://github.com/zaeleus/noodles) that extends FAIDX to support `memmap`-based file I/O for FASTA files.

## Installation

To install from PyPI, execute the following command:

```bash
pip install bionemo-noodles
```

### Compatibility

`bionemo-noodles` has pre-built wheels for Python/Cython `3.10`, `3.11`, and `3.12`, and is compatible with `manylinux_2_28` on `x86_64`.

For a custom build configuration that is not currently supported on PyPI, reach out to: bionemofeedback@nvidia.com

## Usage

An example `torch.utils.data.Dataset` using `NvFaidx` / `bionemo-noodles`:

```
import json
from pathlib import Path

import torch

from bionemo.noodles.nvfaidx import NvFaidx

class SimpleFastaDataset(torch.utils.data.Dataset):

    def __init__(self, fasta_path: Path, tokenizer):
        """Initialize the dataset."""
        super().__init__()
        self.fasta = NvFaidx(fasta_path)
        self.seqids = sorted(self.fasta.keys())
        self.tokenizer = tokenizer

    def write_idx_map(self, output_dir: Path):
        """Write the index map to the output directory."""
        with open(output_dir / "seq_idx_map.json", "w") as f:
            json.dump({seqid: idx for idx, seqid in enumerate(self.seqids)}, f)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.seqids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset."""
        sequence = self.fasta[self.seqids[idx]].sequence().upper()
        tokenized_seq = self.tokenizer.text_to_ids(sequence)
        loss_mask = torch.ones_like(torch.tensor(tokenized_seq, dtype=torch.long), dtype=torch.long)
        return {
            "tokens": torch.tensor(tokenized_seq, dtype=torch.long),
            "position_ids": torch.arange(len(tokenized_seq), dtype=torch.long),
            "seq_idx": torch.tensor(idx, dtype=torch.long),
            "loss_mask": loss_mask,
        }
```

## BioNeMo Framework Ecosystem Development

To install this sub-package locally (with `--editable`):

```
pip install -e .
```

To run unit tests, execute:

```bash
pytest -v .
```

To build wheels for different Python, Linux, and system architecture configurations, run the [BioNeMo Sub-Package GitHub Actions Workflow (bionemo-subpackage-ci.yml)](https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/.github/workflows/bionemo-subpackage-ci.yml)
