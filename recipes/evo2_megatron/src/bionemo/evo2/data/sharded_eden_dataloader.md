# Sharded Eden DataLoader Implementation

## Overview

The `sharded_eden_dataloader.py` implements a dataloader for genomic sequences that uses pre-computed data structures and SQLite databases for efficient data access. This implementation is designed to significantly reduce the computational overhead during training by moving expensive operations to a pre-processing phase.

## Key Features

### 1. Split-Specific Window Databases

- **Sharded**: Uses separate pre-computed window databases for each split:
  - `train_window_db_path`: SQLite database containing window mappings for training data
  - `val_window_db_path`: SQLite database containing window mappings for validation data
  - `test_window_db_path`: SQLite database containing window mappings for test data

### 2. SQLite Database Storage

- **Sharded**: Uses SQLite databases organized by sample:
  - **Per-Sample Sequence Databases**: Each sample has its own SQLite file at `sequence_db_dir/<sample_id>/glm_dataset_<sample_id>.sqlite`
  - **Split-Specific Window Databases**: Pre-computed window mappings stored in separate databases for each data split

### 3. Virtual Window Pre-computation

- **Sharded**: Window mappings are pre-computed from Parquet files and stored in split-specific databases

## Sequence ID Format

Sequence IDs follow a specific format: `BCR__ECT-SAMPLE1__CT1-1`

The sample ID can be extracted using: `extract_sample_id(sequence_id)` which implements `".".join(sequence_id.split("__")[1].split("-")[1:])` (returns `SAMPLE1`)

## Database Schema

### Per-Sample Sequence Database

Each sample has its own SQLite file with the following schema:

```sql
CREATE TABLE sequences (
    contig_id TEXT PRIMARY KEY,
    nt_sequence TEXT NOT NULL
);
```

### Split-Specific Window Database

Each split (train/validation/test) has its own window database:

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE TABLE window_mappings (
    window_idx INTEGER PRIMARY KEY,
    sequence_id TEXT NOT NULL,
    window_in_seq_idx INTEGER NOT NULL
);
CREATE INDEX idx_sequence_id ON window_mappings(sequence_id);
```

The metadata table stores the `window_size` and `stride` parameters used during pre-computation.

## Directory Structure

```
sequence_db_dir/
├── SAMPLE1/
│   └── glm_dataset_SAMPLE1.sqlite
├── SAMPLE2/
│   └── glm_dataset_SAMPLE2.sqlite
├── SAMPLE3/
│   └── glm_dataset_SAMPLE3.sqlite
└── ...

Window databases (separate files):
├── train_windows.db
├── val_windows.db
└── test_windows.db
```

## Usage Example

```python
from bionemo.evo2.run.sharded_eden_dataloader import ShardedEdenDataModule

# Create the data module
data_module = ShardedEdenDataModule(
    sequence_db_dir="path/to/sequence_db_dir",  # Directory containing sample folders
    train_window_db_path="path/to/train_windows.db",
    val_window_db_path="path/to/val_windows.db",
    test_window_db_path="path/to/test_windows.db",
    seq_length=8192,
    micro_batch_size=1,
    global_batch_size=4,
    num_workers=8,
    rc_aug=True,
    use_control_tags=True,
)

# Use with PyTorch Lightning trainer
trainer = pl.Trainer(...)
trainer.fit(model, data_module)
```

## Pre-processing Workflow

### 1. Create Sample Sequence Databases

For each sample, create its SQLite database:

```python
import sqlite3
import os


def create_sample_database(sample_id, sequences, output_dir):
    """Create SQLite database for a single sample."""
    # Create sample directory
    sample_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    # Create database
    db_path = os.path.join(sample_dir, f"glm_dataset_{sample_id}.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute(
        """
        CREATE TABLE sequences (
            contig_id TEXT PRIMARY KEY,
            nt_sequence TEXT NOT NULL
        )
    """
    )

    # Insert sequences for this sample
    for seq_id, sequence in sequences:
        cursor.execute(
            "INSERT INTO sequences (contig_id, nt_sequence) VALUES (?, ?)",
            (seq_id, sequence),
        )

    conn.commit()
    conn.close()


# Example usage
# Group sequences by sample_id
from collections import defaultdict

sequences_by_sample = defaultdict(list)
for seq_id, sequence in all_sequences:  # all_sequences is your data
    sample_id = extract_sample_id(seq_id)
    sequences_by_sample[sample_id].append((seq_id, sequence))

# Create database for each sample
for sample_id, sequences in sequences_by_sample.items():
    create_sample_database(sample_id, sequences, "path/to/sequence_db_dir")
```

### 2. Create Split Data Files

Create Parquet files for each split containing sequence metadata:

```python
import polars as pl

# Create train split Parquet file
train_data = pl.DataFrame(
    {
        "contig_id": ["BCR__ECT-SAMPLE1__CT1-1", "BCR__ECT-SAMPLE1__CT1-2", ...],
        "length": [1500, 2000, ...],  # sequence lengths
    }
)
train_data.write_parquet("train_split.parquet")

# Similarly for validation and test splits
val_data = pl.DataFrame(
    {"contig_id": ["BCR__ECT-SAMPLE2__CT1-1", ...], "length": [1800, ...]}
)
val_data.write_parquet("val_split.parquet")

test_data = pl.DataFrame(
    {"contig_id": ["BCR__ECT-SAMPLE3__CT1-1", ...], "length": [1600, ...]}
)
test_data.write_parquet("test_split.parquet")
```

### 3. Create Window Mappings Databases using CLI

The package includes a CLI tool for pre-computing the window databases:

```bash
# Pre-compute window mappings for training split
python -m bionemo.evo2.run.sharded_eden_dataloader precompute \
    train_split.parquet \
    train_windows.db \
    --window-size 8192 \
    --stride 7992

# Pre-compute window mappings for validation split
python -m bionemo.evo2.run.sharded_eden_dataloader precompute \
    val_split.parquet \
    val_windows.db \
    --window-size 8192 \
    --stride 7992

# Pre-compute window mappings for test split
python -m bionemo.evo2.run.sharded_eden_dataloader precompute \
    test_split.parquet \
    test_windows.db \
    --window-size 8192 \
    --stride 7992
```

## Implementation Details

### Key Components

1. **ShardedEdenDataModule**:

   - Uses separate window databases for each split (train/val/test)
   - Manages per-sample SQLite file paths
   - Creates datasets with directory and database paths
   - Handles distributed training setup with Megatron integration

2. **ShardedEdenDataset**:

   - Automatically discovers sample SQLite files from directory structure
   - Maps sequence IDs to appropriate sample databases using `extract_sample_id()`
   - Pre-opens all database connections for performance
   - Attaches window database to each sequence connection for efficient JOINs
   - Implements sequence caching with connection pooling
   - Maintains compatibility with original tokenization and formatting logic
   - Optional window access logging for performance analysis

3. **CLI Tool**:

   - `precompute`: Creates window databases from Parquet files

### Advanced Features

#### Window Access Logging

Enable detailed logging of window access patterns:

```python
dataset = ShardedEdenDataset(
    # ... other parameters ...
    log_windows=True,
    log_dir="sequence_logs",
)
```

This creates CSV logs tracking which windows are accessed, useful for analyzing data loading patterns.

#### Connection Management

- All database connections are pre-opened during initialization for performance
- Database connections are pooled and reused per sample
- Sequence data is fetched on-demand using SQL SUBSTR for memory efficiency
- Position IDs are shared across instances to reduce memory usage
- Connections are properly closed when dataset is destroyed

#### Metadata Validation

The implementation validates that window databases were created with compatible parameters:

- Checks stored `window_size` matches dataset `seq_length`
- Checks stored `stride` matches dataset `stride`
- Provides clear error messages for mismatches

### Error Handling

- Validates sample SQLite files exist during initialization
- Handles missing sequences gracefully with informative error messages
- Ensures proper cleanup of database connections
- Provides detailed debugging information for database issues
- Validates Parquet file schema during pre-computation
