# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# high performance implementation of the EdenDataModule, assuming some items are pre-computed + sharded fasta files and fasta index files.
# Contributed by: BaseCamp Research https://basecamp-research.com/ https://github.com/NVIDIA/bionemo-framework/pull/1091
import argparse
import csv
import os
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import polars as pol
import torch
import torch.distributed as dist
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from nemo.utils.import_utils import safe_import
from torch.utils.data import Dataset, default_collate

from bionemo.core.data.multi_epoch_dataset import (
    IdentityMultiEpochDatasetWrapper,
    MultiEpochDatasetResampler,
)


# -----------------------------------------------------------------------------
# Configurable column names
# -----------------------------------------------------------------------------
# Column name for IDs in input data (e.g., parquet) and in shard SQLite tables
SEQUENCE_ID_COLUMN_NAME = "contig_id"
# Column name for sequence lengths in input data (e.g., parquet)
SEQUENCE_LENGTH_COLUMN_NAME = "length"
# Column name for nucleotide/amino-acid sequence in shard SQLite tables
SEQUENCE_COLUMN_NAME = "nt_sequence"

_, HAVE_TE = safe_import("transformer_engine")

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def extract_sample_id(sequence_id: str) -> str:
    """Extract sample ID from sequence ID format: BCR__EXT-SAMPLE1__CT1-1."""
    parts = sequence_id.split("__")[1].split("-")[1:]
    return ".".join(parts)


class ShardedEdenDataModule(pl.LightningDataModule):
    """High-performance DataModule that uses pre-computed splits and SQLite databases.

    Key differences from EdenDataModule:
    - Train/val/test splits are loaded from numpy array files
    - Sequence data stored in per-sample SQLite databases
    - Virtual window mappings pre-computed and stored in separate SQLite database
    """

    def __init__(
        self,
        sequence_db_dir: str,  # Directory containing sample SQLite files
        train_window_db_path: str,  # Path to the pre-computed DB for the training split
        val_window_db_path: str,  # Path to the pre-computed DB for the validation split
        test_window_db_path: str,  # Path to the pre-computed DB for the test split
        seq_length: int = 8192,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 1,
        global_batch_size: int = 4,
        rampup_batch_size: Optional[List[int]] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        rc_aug: bool = False,
        stride: int = 7992,
        window_min_length_threshold: Optional[int] = None,
        use_control_tags: bool = False,
        seed: int = 42,
        num_epochs: int = 1,
        log_windows: bool = False,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the ShardedEdenDataModule. See sub-packages/bionemo-evo2/src/bionemo/evo2/data/sharded_eden_dataloader.md for how to prepare the input data."""
        super().__init__()
        self.sequence_db_dir = sequence_db_dir
        self.train_window_db_path = train_window_db_path
        self.val_window_db_path = val_window_db_path
        self.test_window_db_path = test_window_db_path
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask or not HAVE_TE
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else 7992
        # Minimum effective window length used at precomputation time. If None or 0, disabled.
        self.window_min_length_threshold = int(window_min_length_threshold) if window_min_length_threshold else 0
        self.use_control_tags = use_control_tags
        self.init_global_step = 0
        self.seed = seed
        self.num_epochs = num_epochs
        self.log_windows = log_windows
        self.log_dir = log_dir

        if tokenizer is None:
            self.tokenizer = get_nmt_tokenizer(
                "megatron",
                "GPT2BPETokenizer",
                vocab_file=vocab_file,
                merges_file=merges_file,
            )
        else:
            self.tokenizer = tokenizer

        # Megatron sampler
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def build(
        self,
        trainer_max_steps: int,
        trainer_val_check_interval: Union[int, float],
        trainer_limit_val_batches: Union[int, float],
        trainer_limit_test_batches: Union[int, float],
    ):
        """Build the datasets using pre-computed, split-specific window databases."""
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Creating datasets from pre-computed, split-specific window databases.")

        # Create datasets wrapped with epoch-based resampler
        self._train_ds = self._create_epoch_wrapped_sharded_eden_dataset(
            window_db_path=self.train_window_db_path,
            split="train",
            shuffle=True,
        )

        self._validation_ds = self._create_epoch_wrapped_sharded_eden_dataset(
            window_db_path=self.val_window_db_path,
            split="validation",
            shuffle=False,
        )

        self._test_ds = self._create_epoch_wrapped_sharded_eden_dataset(
            window_db_path=self.test_window_db_path,
            split="test",
            shuffle=False,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"Dataset windows: Train={len(self._train_ds)}, Val={len(self._validation_ds)}, Test={len(self._test_ds)}"
            )

    def setup(self, stage: str = "") -> None:
        """Setup the data module."""
        assert hasattr(self, "trainer") and self.trainer is not None, (
            "Setup should be completed when trainer and config are attached."
        )

        self.build(
            trainer_max_steps=self.trainer.max_steps,
            trainer_val_check_interval=self.trainer.val_check_interval,
            trainer_limit_val_batches=self.trainer.limit_val_batches,
            trainer_limit_test_batches=self.trainer.limit_test_batches,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get the train dataloader."""
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get the validation dataloader."""
        return self._create_dataloader(self._validation_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get the test dataloader."""
        return self._create_dataloader(self._test_ds, mode="test")

    def _create_dataloader(self, dataset, mode, **kwargs) -> WrappedDataLoader:
        assert hasattr(self, "trainer") and self.trainer is not None, (
            "Trainer must be attached before creating dataloaders."
        )
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        dataloader = WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=getattr(dataset, "collate_fn", default_collate),
            **kwargs,
        )
        return dataloader

    def _create_epoch_wrapped_sharded_eden_dataset(
        self,
        *,
        window_db_path: str,
        split: str,
        shuffle: bool,
    ) -> MultiEpochDatasetResampler:
        """Instantiate `ShardedEdenDataset` and wrap it with `MultiEpochDatasetResampler`.

        By default, `num_epochs=1`, so the wrapped dataset length equals the base dataset length.
        """
        base_dataset = ShardedEdenDataset(
            tokenizer=self.tokenizer,
            sequence_db_dir=self.sequence_db_dir,
            window_db_path=window_db_path,
            seq_length=self.seq_length,
            create_attention_mask=self.create_attention_mask,
            stride=self.stride,
            window_min_length_threshold=self.window_min_length_threshold,
            rc_aug=self.rc_aug,
            use_control_tags=self.use_control_tags,
            split=split,
            log_windows=self.log_windows,
            log_dir=self.log_dir,
        )

        wrapped = MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(base_dataset),
            num_epochs=self.num_epochs,
            shuffle=shuffle,
            seed=self.seed,
        )
        return wrapped

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint."""
        consumed_samples = self.data_sampler.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        return {"consumed_samples": consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        try:
            from megatron.core.num_microbatches_calculator import (
                update_num_microbatches,
            )
        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict["consumed_samples"]
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1

    def reconfigure_limit_batches(self):
        """Reconfigure trainer.limit_train_batches and trainer.limit_val_batches."""
        self._reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_ds, "train")
        self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_ds, "val")

    def _reconfigure_limit_batches(self, limit_batches, dataloader, mode):
        """Reconfigure limit_batches for distributed training."""
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches
        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        if isinstance(limit_batches, int):
            limit_batches *= get_num_microbatches()
        else:
            assert isinstance(limit_batches, float)
            if limit_batches == 0.0 or dataloader is None:
                return

            dl_len_in_micro_batches = len(dataloader)
            if len(dataloader) != float("inf"):
                if limit_batches == 1.0:
                    limit_batches = dl_len_in_micro_batches
                else:
                    limit_micro_batches = int(dl_len_in_micro_batches * limit_batches)
                    if limit_micro_batches == 0 and limit_batches > 0.0:
                        min_percentage = 1.0 / len(dataloader)
                        raise ValueError(
                            f"You requested to check {limit_batches} of the val_dataloader but"
                            f" {limit_batches} * {len(dataloader)} < 1. Please increase the"
                            f" `limit_val_batches` argument. Try at least"
                            f" `limit_val_batches={min_percentage}`"
                        )
                    if limit_micro_batches < get_num_microbatches():
                        limit_batches = get_num_microbatches()
                    else:
                        limit_batches = limit_batches - limit_batches % get_num_microbatches()

        if mode == "train":
            self.trainer.limit_train_batches = limit_batches
        else:
            self.trainer.limit_val_batches = limit_batches


class ShardedEdenDataset(Dataset):
    """High-performance Dataset that uses SQLite databases for sequence storage and window mapping. Assumes that the window_db_path points to a database pre-computed for a specific data split (e.g., train, validation, or test)."""

    def __init__(
        self,
        tokenizer,
        sequence_db_dir: str,
        window_db_path: str,
        seq_length: int,
        create_attention_mask: bool = False,
        rc_aug: bool = False,
        stride: Optional[int] = 7992,
        window_min_length_threshold: Optional[int] = None,
        use_control_tags: bool = False,
        split: str = "train",
        log_windows: bool = False,
        log_dir: Optional[str] = None,
        skip_stats: bool = True,
    ) -> None:
        """Initialize the ShardedEdenDataset."""
        super().__init__()
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.sequence_db_dir = sequence_db_dir
        self.window_db_path = window_db_path
        self.create_attention_mask = create_attention_mask
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else 7992
        self.window_min_length_threshold = int(window_min_length_threshold) if window_min_length_threshold else 0
        self.use_control_tags = use_control_tags
        self.split = split
        self.skip_stats = skip_stats
        # Window access logging setup (lazy init in __getitem__)
        self.log_windows = log_windows
        # Remember desired log directory for lazy init in worker processes
        self._log_dir = log_dir

        # Create mapping from sample_id to SQLite file path
        self._create_sample_db_mapping()

        # Pre-open all database connections for performance
        self._open_all_sequence_dbs()

        # Validates metadata and sets up the dataset
        self._validate_and_setup_db()

        # Prepare control-tag IDs if needed
        if self.use_control_tags:
            self._prepare_control_tags()

        # Attention mask and position ids
        if create_attention_mask:
            self.attention_mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0) < 0.5

        # Shared position_ids for memory efficiency
        if not hasattr(ShardedEdenDataset, "_position_ids") or ShardedEdenDataset._position_ids.size(0) != seq_length:
            ShardedEdenDataset._position_ids = torch.arange(seq_length, dtype=torch.int64)
        self.position_ids = ShardedEdenDataset._position_ids

        # Counter for periodic commits if logging is enabled
        if self.log_windows:
            self._log_counter = 0

    def _open_all_sequence_dbs(self):
        """Open all sequence database files ahead of time and attach the window database for efficient cross-database queries."""
        self.db_connections = {}
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Pre-opening {len(self.sample_db_mapping)} sequence database files...")

        for sample_id, db_path in self.sample_db_mapping.items():
            try:
                # URI=true allows for read-only connections if needed and more options
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                self.db_connections[sample_id] = conn
            except sqlite3.Error as e:
                logging.error(f"Failed to open/attach database for sample {sample_id} at {db_path}: {e}")
                raise

    def _create_sample_db_mapping(self):
        """Create mapping from sample ID to SQLite file path."""
        self.sample_db_mapping = {}

        # Scan the directory for sample SQLite files
        db_dir = Path(self.sequence_db_dir)
        for sample_dir in db_dir.iterdir():
            if sample_dir.is_dir():
                sample_id = sample_dir.name
                db_file = sample_dir / f"glm_dataset_{sample_id}.sqlite"
                if db_file.exists():
                    self.sample_db_mapping[sample_id] = str(db_file)

        if not self.sample_db_mapping:
            raise ValueError(f"No SQLite files found in {self.sequence_db_dir}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {len(self.sample_db_mapping)} sample SQLite files")

    def _validate_and_setup_db(self):
        """Connects to the window database, validates its metadata, and computes the length of the dataset for the current split."""
        self.window_db_conn = sqlite3.connect(f"file:{self.window_db_path}?mode=ro", uri=True)
        cursor = self.window_db_conn.cursor()

        # Validate metadata
        try:
            cursor.execute("SELECT key, value FROM metadata")
            db_meta = dict(cursor.fetchall())

            if "window_size" not in db_meta or "stride" not in db_meta:
                raise ValueError("Database metadata is missing 'window_size' or 'stride' keys.")

            db_window_size = int(db_meta["window_size"])
            db_stride = int(db_meta["stride"])
            db_min_len_raw = db_meta.get("window_min_length_threshold")
            db_min_len = int(db_min_len_raw) if db_min_len_raw is not None else None

            if db_window_size != self.seq_length or db_stride != self.stride:
                raise ValueError(
                    f"Database metadata mismatch! "
                    f"DB created with window_size={db_window_size}, stride={db_stride}. "
                    f"Dataset configured with seq_length={self.seq_length}, stride={self.stride}. "
                    f"Please re-run pre-computation or check your config."
                )

            # Validate presence and value of window_min_length_threshold only if enabled
            if self.window_min_length_threshold and self.window_min_length_threshold > 0:
                if db_min_len is None:
                    raise ValueError(
                        "Database metadata is missing 'window_min_length_threshold'. "
                        "Please re-run the pre-computation script with an updated version to populate this key."
                    )
                if db_min_len != self.window_min_length_threshold:
                    raise ValueError(
                        f"Database metadata mismatch for window_min_length_threshold! "
                        f"DB created with window_min_length_threshold={db_min_len}. "
                        f"Dataset configured with window_min_length_threshold={self.window_min_length_threshold}. "
                        f"Please re-run pre-computation or align the configuration."
                    )
            else:
                # Case: DB is pruned but runtime threshold is not set (> 0)
                if db_min_len is not None and int(db_min_len) > 0:
                    raise ValueError(
                        f"Window DB indicates pruning was applied (window_min_length_threshold={db_min_len}), "
                        "but the current configuration does not set --window-min-length-threshold (> 0). "
                        "Please set the argument to match the DB or use an unpruned database."
                    )
        except sqlite3.OperationalError:
            raise ValueError(
                f"Could not find `metadata` table in {self.window_db_path}. "
                "Please ensure the database was created with a recent version of the pre-computation script."
            )

        # Require modern metadata keys
        if "total_windows" not in db_meta or "distinct_sequences" not in db_meta:
            raise ValueError(
                "Database metadata must contain 'total_windows' and 'distinct_sequences'. "
                "Please re-run the pre-computation script to create an up-to-date window database."
            )

        # Read counts directly from metadata
        self.length = int(db_meta["total_windows"])

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.length} windows for {self.split} split in {self.window_db_path}.")

        # Distinct sequences directly from metadata
        self.distinct_sequences = int(db_meta["distinct_sequences"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.distinct_sequences} distinct sequences.")

    def _prepare_control_tags(self):
        """Prepare control tag IDs for sequences."""
        self.ctrl_ids_map = {}

        # Query the split-specific DB for the sequences it contains
        cursor = self.window_db_conn.cursor()
        unique_sequence_ids = [row[0] for row in cursor.execute("SELECT DISTINCT sequence_id FROM window_mappings")]

        # Create control tags for unique sequence IDs in this split
        for seq_id in unique_sequence_ids:
            # Extract meaningful part from sequence ID for control tag
            ctrl_name = seq_id.split("__")[0] if "__" in seq_id else seq_id
            self.ctrl_ids_map[seq_id] = self.tokenizer.text_to_ids(f"<ctrl_{ctrl_name.lower()}>")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def _get_db_connection(self, sample_id: str) -> sqlite3.Connection:
        """Get a pre-opened database connection for a sample."""
        conn = self.db_connections.get(sample_id)
        if conn is None:
            # This should not happen if _open_all_sequence_dbs was called
            # and the sample_id is valid.
            raise ValueError(f"No pre-opened SQLite connection found for sample {sample_id}")
        return conn

    def reverse_complement(self, seq: str) -> str:
        """Compute reverse complement of a sequence."""
        cmap = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(cmap.get(b, b) for b in reversed(seq))

    def __getitem__(self, idx: np.int64) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset with length {self.length}")

        # The dataloader index `idx` is now the `window_idx` in this split-specific DB.
        # Step 1: Get the sequence_id and other info from the window DB.
        window_cursor = self.window_db_conn.cursor()
        res = window_cursor.execute(
            "SELECT sequence_id, window_in_seq_idx FROM window_mappings WHERE window_idx = ?",
            (int(idx),),
        ).fetchone()

        if res is None:
            # run PRAGMA database_list; to check exactly which DBs are open
            # and which ones are not
            current_dbs = self.window_db_conn.execute("PRAGMA database_list;").fetchall()

            raise IndexError(
                f"Window index {idx} which is a {type(idx)} was not found in the database {current_dbs}, which is unexpected."
            )

        sequence_id, window_in_seq_idx = res

        # Log window access if enabled
        if self.log_windows:
            # Ensure logger is initialised in the current process (e.g., after DataLoader forks)
            if not hasattr(self, "_log_writer"):
                self._init_window_logger(self._log_dir)

            # Derive sample_id for logging independent of DB connection logic
            try:
                sample_id_for_log = extract_sample_id(sequence_id)
            except Exception:
                sample_id_for_log = "unknown"

            # Synchronously write CSV row (no flush here; only on cleanup)
            row = [
                int(idx),
                sequence_id,
                sample_id_for_log,
                int(window_in_seq_idx),
                int(self._rank),
                int(time.time_ns()),
            ]
            self._log_writer.writerow(row)
            self._log_file.flush()

        # if there is only one DB connection, use it directly
        if len(self.db_connections) == 1:
            conn = next(iter(self.db_connections.values()))
            cursor = conn.cursor()
            sample_id = None
        else:
            # otherwise, we need to get the sample_id from the sequence_id
            # and get the DB connection for that sample
            sample_id = extract_sample_id(sequence_id)
            conn = self._get_db_connection(sample_id)
            cursor = conn.cursor()

        # Calculate window position within sequence (0-based for Python, +1 for SQLite SUBSTR)
        start_pos = window_in_seq_idx * self.stride

        # Build token window
        ctrl_ids = self.ctrl_ids_map.get(sequence_id, []) if self.use_control_tags else []
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        sep_id = self.tokenizer._sep_id
        pad_id = self.tokenizer.pad_id

        header = [bos_id] + ctrl_ids + [sep_id]
        footer = [eos_id]
        special_tokens_count = len(header) + len(footer)
        eff_len = self.seq_length - special_tokens_count

        # ------------------------------------------------------------------
        # Retrieve the subsequence directly in SQL to avoid loading the
        # full contig into Python memory.
        # SQLite SUBSTR is 1-indexed, so we add 1 to start_pos.
        # ------------------------------------------------------------------
        subseq_query = (
            f"SELECT substr({SEQUENCE_COLUMN_NAME}, ?, ?) FROM sequences WHERE {SEQUENCE_ID_COLUMN_NAME} = ?"
        )
        result = cursor.execute(
            subseq_query,
            (start_pos + 1, eff_len, sequence_id),
        ).fetchone()

        if result is None or result[0] is None:
            raise ValueError(f"Sequence ID {sequence_id} not found in database for sample {sample_id}")

        seq = result[0].upper()

        # Apply reverse complement augmentation if enabled
        if self.rc_aug and np.random.rand() > 0.5:
            seq = self.reverse_complement(seq)

        # Tokenize
        token_ids = header + self.tokenizer.text_to_ids(seq) + footer

        # Pad/trim
        if len(token_ids) < self.seq_length:
            token_ids += [pad_id] * (self.seq_length - len(token_ids))
        else:
            token_ids = token_ids[: self.seq_length]

        tokens = torch.tensor(token_ids, dtype=torch.int64)

        # Flatten ctrl_ids and create special_ids list
        flat_ctrl_ids = []
        if isinstance(ctrl_ids, list):
            for item in ctrl_ids:
                if isinstance(item, list):
                    flat_ctrl_ids.extend(item)
                else:
                    flat_ctrl_ids.append(item)

        special_ids_list = [bos_id, eos_id, sep_id, pad_id] + flat_ctrl_ids
        special_ids = torch.tensor(special_ids_list, dtype=torch.int64)

        # Create labels for next token prediction
        labels = tokens.clone()
        labels[:-1] = tokens[1:]
        labels[-1] = pad_id

        # Create loss mask
        loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        loss_mask[torch.isin(labels, special_ids)] = 0

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": self.position_ids,
        }
        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask

        return batch

    def collate_fn(self, batch):
        """Collate a batch of items into a single dictionary."""
        return default_collate(batch)

    def __del__(self):
        """Close all database connections when the dataset is destroyed."""
        # Close window mapping DB
        if hasattr(self, "window_db_conn") and self.window_db_conn:
            self.window_db_conn.close()

        # Close all sequence shard DBs
        if hasattr(self, "db_connections"):
            for conn in self.db_connections.values():
                conn.close()

        if hasattr(self, "_log_file") and self._log_file:
            try:
                self._log_file.flush()
            except Exception:
                pass
            try:
                self._log_file.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Logging helper methods
    # ------------------------------------------------------------------

    def _init_window_logger(self, log_dir: Optional[str] = None):
        """Initialise CSV file for window access logging."""
        import uuid

        rank = dist.get_rank() if dist.is_initialized() else 0
        self._rank = rank
        log_uuid = str(uuid.uuid4())
        base_dir = Path(log_dir) if log_dir else Path(os.getcwd())
        base_dir = base_dir.resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        split_tag = getattr(self, "split", "unknown")
        csv_path = (base_dir / f"window_access_{split_tag}_rank{rank}_{log_uuid[:8]}.csv").resolve()
        # raise an error if the file already exists
        if csv_path.exists():
            raise FileExistsError(
                f"File {csv_path} already exists, this should only happen on a uuid conflict and should be extremely rare"
            )
        self._log_file_path = str(csv_path)

        # Open CSV file in append mode and write header
        self._log_file = open(self._log_file_path, mode="a", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(
            [
                "window_idx",
                "sequence_id",
                "sample_id",
                "window_in_seq_idx",
                "rank",
                "access_ts",
            ]
        )

        print(f"Window access logger initialised at {self._log_file_path}")


def compute_num_windows(seq_len: int, window_size: int = 8192, stride: int = 7992) -> int:
    """Helper method to compute number of windows for a sequence."""
    if seq_len < window_size:
        return 1
    else:
        return 1 + (seq_len - window_size) // stride


def precompute_window_database(
    split_parquet_file: str,
    output_window_db: str,
    window_size: int = 8192,
    stride: int = 7992,
    window_min_length_threshold: int = 0,
):
    """Pre-compute window mappings for a split using a Parquet file. The Parquet file must contain ID and length columns as configured by `SEQUENCE_ID_COLUMN_NAME` and `SEQUENCE_LENGTH_COLUMN_NAME`.

    The output database will contain two tables:
    1. `metadata`: Stores the window_size and stride used for creation.
    2. `window_mappings`: A mapping of window_idx to sequence_id and the
       relative index of that window within the sequence.

    Args:
        split_parquet_file: Path to a Parquet file with ID and length columns.
        output_window_db: Path to output window mapping database
        window_size: Window size (default: 8192)
        stride: Stride between windows (default: 7992)
        window_min_length_threshold: Minimum length of windows to include (default: 0)
    """
    print(f"Creating window database at {output_window_db} from {split_parquet_file}")
    print(
        f"Using window_size={window_size}, stride={stride}, window_min_length_threshold={window_min_length_threshold}"
    )

    # Load sequence data from Parquet file
    try:
        df = pol.read_parquet(split_parquet_file)
    except Exception as e:
        raise IOError(f"Failed to read Parquet file at {split_parquet_file}") from e

    # Validate columns
    if SEQUENCE_ID_COLUMN_NAME not in df.columns or SEQUENCE_LENGTH_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Parquet file {split_parquet_file} must contain '"
            f"{SEQUENCE_ID_COLUMN_NAME}' and '{SEQUENCE_LENGTH_COLUMN_NAME}' columns."
        )

    # Sort by ID to ensure deterministic window ordering
    df = df.sort(SEQUENCE_ID_COLUMN_NAME)

    conn = sqlite3.connect(output_window_db)
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # High-performance SQLite settings for bulk insert (100M+ rows)
    # ------------------------------------------------------------------
    cursor.execute("PRAGMA journal_mode=OFF;")  # disable rollback journal
    cursor.execute("PRAGMA synchronous=OFF;")  # no fsync
    cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")  # single-writer, avoids lock churn
    cursor.execute("PRAGMA temp_store=MEMORY;")  # temp tables in RAM
    cursor.execute("PRAGMA cache_size=-1048576;")  # ~1 GB cache (negative => KiB)
    # page_size and mmap_size can only be set before DB creation; assume defaults suffice.

    # Drop old tables if they exist
    cursor.execute("DROP TABLE IF EXISTS window_mappings")
    cursor.execute("DROP TABLE IF EXISTS metadata")

    # Create metadata table
    cursor.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)
    cursor.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        [
            ("window_size", window_size),
            ("stride", stride),
            (
                "window_min_length_threshold",
                int(window_min_length_threshold) if window_min_length_threshold else 0,
            ),
        ],
    )

    # Create window mappings table
    cursor.execute("""
        CREATE TABLE window_mappings (
            window_idx INTEGER PRIMARY KEY,
            sequence_id TEXT NOT NULL,
            window_in_seq_idx INTEGER NOT NULL
        )
    """)
    conn.commit()

    total_sequences = 0
    global_window_idx = 0
    batch_size = 20000
    batch = []
    skipped_windows = 0

    for seq_id, seq_len in df.select([SEQUENCE_ID_COLUMN_NAME, SEQUENCE_LENGTH_COLUMN_NAME]).iter_rows():
        num_windows = compute_num_windows(seq_len, window_size, stride)

        windows_added_for_seq = 0
        for i in range(num_windows):
            # Determine effective window length at this index
            start_pos = i * stride if seq_len >= window_size else 0
            remaining = max(0, seq_len - start_pos)
            effective_window_len = min(window_size, remaining)

            # Skip windows that are shorter than threshold (if enabled)
            if window_min_length_threshold and effective_window_len < window_min_length_threshold:
                skipped_windows += 1
                continue

            batch.append((global_window_idx, seq_id, i))
            global_window_idx += 1
            windows_added_for_seq += 1

        # Only count sequences that contributed at least one retained window
        if windows_added_for_seq > 0:
            total_sequences += 1

        if len(batch) >= batch_size:
            cursor.executemany(
                "INSERT INTO window_mappings (window_idx, sequence_id, window_in_seq_idx) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
            batch = []
            print(f"Processed {global_window_idx} windows... (skipped {skipped_windows})")

    if batch:
        cursor.executemany(
            "INSERT INTO window_mappings (window_idx, sequence_id, window_in_seq_idx) VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()

    print("Creating index on sequence_id for faster lookups...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequence_id ON window_mappings(sequence_id)")

    # Persist total number of windows in metadata for fast retrieval
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("total_windows", int(global_window_idx)),
    )
    # Persist number of distinct sequences as well
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("distinct_sequences", int(total_sequences)),
    )

    conn.commit()
    conn.close()

    print(f"Finished. Found {total_sequences} sequences and {global_window_idx} total windows.")
    if window_min_length_threshold and skipped_windows > 0:
        print(f"Skipped {skipped_windows} windows due to window_min_length_threshold={window_min_length_threshold}.")


def main():
    """CLI for sharded Eden dataloader utilities."""
    parser = argparse.ArgumentParser(description="Utilities for sharded Eden dataloader: precompute window mappings.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Precompute subcommand
    precompute_parser = subparsers.add_parser("precompute", help="Pre-compute window mappings from a Parquet file")
    precompute_parser.add_argument(
        "split_parquet_file",
        type=str,
        help="Path to a Parquet file with sequence_id and length columns.",
    )
    precompute_parser.add_argument("output_window_db", type=str, help="Path to output window mapping database")
    precompute_parser.add_argument("--window-size", type=int, default=8192, help="Window size (default: 8192)")
    precompute_parser.add_argument(
        "--stride",
        type=int,
        default=7992,
        help="Stride between windows (default: 7992)",
    )
    precompute_parser.add_argument(
        "--window-min-length-threshold",
        type=int,
        default=0,
        help=("If > 0, skip sequences shorter than this length when precomputing windows. Defaults to 0 (disabled)."),
    )

    args = parser.parse_args()

    if args.command == "precompute":
        precompute_window_database(
            args.split_parquet_file,
            args.output_window_db,
            args.window_size,
            args.stride,
            args.window_min_length_threshold,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
