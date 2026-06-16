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

"""High-performance SQLite-backed genomic dataset and window pre-computation.

Contributed by BaseCamp Research: https://basecamp-research.com/
https://github.com/NVIDIA/bionemo-framework/pull/1091

This module is used by multiple recipes via ``bionemo.common``.
**It must not import megatron-core, megatron-bridge, or NeMo.**
"""

import argparse
import csv
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import polars as pol
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, default_collate

from bionemo.common.data.basecamp.utils import (
    SEQUENCE_COLUMN_NAME,
    SEQUENCE_ID_COLUMN_NAME,
    SEQUENCE_LENGTH_COLUMN_NAME,
    extract_sample_id,
)


logger = logging.getLogger(__name__)


class ShardedEdenDataset(Dataset):
    """High-performance Dataset backed by SQLite databases for sequence storage and window mapping.

    Assumes that ``window_db_path`` points to a database pre-computed for a
    specific data split (e.g., train, validation, or test).
    """

    def __init__(
        self,
        tokenizer: Any,
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
        include_eos: bool = True,
        include_bos: bool = True,
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
        self.log_windows = log_windows
        self._log_dir = log_dir
        self.include_eos = include_eos
        self.include_bos = include_bos

        self._create_sample_db_mapping()
        self._open_all_sequence_dbs()
        self._validate_and_setup_db()

        if self.use_control_tags:
            self._prepare_control_tags()

        if create_attention_mask:
            self.attention_mask = torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0) < 0.5

        if not hasattr(ShardedEdenDataset, "_position_ids") or ShardedEdenDataset._position_ids.size(0) != seq_length:
            ShardedEdenDataset._position_ids = torch.arange(seq_length, dtype=torch.int64)
        self.position_ids = ShardedEdenDataset._position_ids

        if self.log_windows:
            self._log_counter = 0

    def _open_all_sequence_dbs(self):
        """Open all sequence database files ahead of time."""
        self.db_connections = {}
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Pre-opening {len(self.sample_db_mapping)} sequence database files...")

        for sample_id, db_path in self.sample_db_mapping.items():
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                self.db_connections[sample_id] = conn
            except sqlite3.Error as e:
                logger.error(f"Failed to open/attach database for sample {sample_id} at {db_path}: {e}")
                raise

    def _create_sample_db_mapping(self):
        """Create mapping from sample ID to SQLite file path."""
        self.sample_db_mapping = {}

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
        """Connect to the window database, validate metadata, and compute dataset length."""
        self.window_db_conn = sqlite3.connect(f"file:{self.window_db_path}?mode=ro", uri=True)
        cursor = self.window_db_conn.cursor()

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

        if "total_windows" not in db_meta or "distinct_sequences" not in db_meta:
            raise ValueError(
                "Database metadata must contain 'total_windows' and 'distinct_sequences'. "
                "Please re-run the pre-computation script to create an up-to-date window database."
            )

        self.length = int(db_meta["total_windows"])

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.length} windows for {self.split} split in {self.window_db_path}.")

        self.distinct_sequences = int(db_meta["distinct_sequences"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Found {self.distinct_sequences} distinct sequences.")

    def _prepare_control_tags(self):
        """Prepare control tag IDs for sequences."""
        self.ctrl_ids_map = {}

        cursor = self.window_db_conn.cursor()
        unique_sequence_ids = [row[0] for row in cursor.execute("SELECT DISTINCT sequence_id FROM window_mappings")]

        for seq_id in unique_sequence_ids:
            ctrl_name = seq_id.split("__")[0] if "__" in seq_id else seq_id
            if hasattr(self.tokenizer, "tokenize"):
                ctrl_ids = self.tokenizer.tokenize(f"<ctrl_{ctrl_name.lower()}>")
            else:
                ctrl_ids = self.tokenizer.text_to_ids(f"<ctrl_{ctrl_name.lower()}>")
            self.ctrl_ids_map[seq_id] = ctrl_ids

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def _get_db_connection(self, sample_id: str) -> sqlite3.Connection:
        """Get a pre-opened database connection for a sample."""
        conn = self.db_connections.get(sample_id)
        if conn is None:
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

        window_cursor = self.window_db_conn.cursor()
        res = window_cursor.execute(
            "SELECT sequence_id, window_in_seq_idx FROM window_mappings WHERE window_idx = ?",
            (int(idx),),
        ).fetchone()

        if res is None:
            current_dbs = self.window_db_conn.execute("PRAGMA database_list;").fetchall()
            raise IndexError(
                f"Window index {idx} which is a {type(idx)} was not found in the database {current_dbs}, "
                "which is unexpected."
            )

        sequence_id, window_in_seq_idx = res

        if self.log_windows:
            if not hasattr(self, "_log_writer"):
                self._init_window_logger(self._log_dir)

            try:
                sample_id_for_log = extract_sample_id(sequence_id)
            except Exception:
                sample_id_for_log = "unknown"

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

        if len(self.db_connections) == 1:
            conn = next(iter(self.db_connections.values()))
            cursor = conn.cursor()
            sample_id = None
        else:
            sample_id = extract_sample_id(sequence_id)
            conn = self._get_db_connection(sample_id)
            cursor = conn.cursor()

        start_pos = window_in_seq_idx * self.stride

        ctrl_ids = self.ctrl_ids_map.get(sequence_id, []) if self.use_control_tags else []
        bos_id = self.bos_id
        eos_id = self.eos_id
        sep_id = self.sep_id
        pad_id = self.pad_id
        if self.use_control_tags:
            header = [bos_id, *ctrl_ids, sep_id]
            footer = [eos_id] if self.include_eos else []
            special_tokens_count = len(header) + len(footer)
            eff_len = self.seq_length - special_tokens_count
        else:
            header = [bos_id] if self.include_bos else []
            footer = [eos_id] if self.include_eos else []
            special_tokens_count = len(header) + len(footer)
            eff_len = self.seq_length - special_tokens_count

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

        if self.rc_aug and np.random.rand() > 0.5:  # noqa: NPY002
            seq = self.reverse_complement(seq)

        if hasattr(self.tokenizer, "tokenize"):
            token_ids = header + self.tokenizer.tokenize(seq) + footer
        else:
            token_ids = header + self.tokenizer.text_to_ids(seq) + footer

        if len(token_ids) < self.seq_length:
            token_ids += [pad_id] * (self.seq_length - len(token_ids))
        else:
            token_ids = token_ids[: self.seq_length]

        tokens = torch.tensor(token_ids, dtype=torch.int64)

        flat_ctrl_ids = []
        if isinstance(ctrl_ids, list):
            for item in ctrl_ids:
                if isinstance(item, list):
                    flat_ctrl_ids.extend(item)
                else:
                    flat_ctrl_ids.append(item)

        special_ids_list = [bos_id, eos_id, sep_id, pad_id, *flat_ctrl_ids]
        special_ids = torch.tensor(special_ids_list, dtype=torch.int64)

        labels = tokens.clone()
        labels[:-1] = tokens[1:]
        labels[-1] = pad_id

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

    @property
    def bos_id(self) -> int:
        """Get the beginning of sequence token ID."""
        return self.tokenizer.bos_id

    @property
    def eos_id(self) -> int:
        """Get the end of sequence token ID."""
        return self.tokenizer.eos_id

    @property
    def sep_id(self) -> int:
        """Get the separator token ID."""
        sep_id = getattr(self.tokenizer, "_sep_id", None)
        if sep_id is None:
            if hasattr(self.tokenizer, "tokenize"):
                sep_id = self.tokenizer.tokenize("<SEP>")
            else:
                sep_id = self.tokenizer.text_to_ids("<SEP>")
            if len(sep_id) == 1:
                sep_id = sep_id[0]
            else:
                sep_id = None
        if sep_id is None:
            return self.eos_id
        return sep_id

    @property
    def pad_id(self) -> int:
        """Get the padding token ID."""
        pad_id = getattr(self.tokenizer, "pad_id", None)
        if pad_id is None:
            if hasattr(self.tokenizer, "tokenize"):
                pad_id = self.tokenizer.tokenize("<PAD>")
            else:
                pad_id = self.tokenizer.text_to_ids("<PAD>")
            if len(pad_id) == 1:
                pad_id = pad_id[0]
            else:
                pad_id = None
        if pad_id is None:
            return self.eos_id
        return pad_id

    def __del__(self):
        """Close all database connections when the dataset is destroyed."""
        if hasattr(self, "window_db_conn") and self.window_db_conn:
            self.window_db_conn.close()

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
        if csv_path.exists():
            raise FileExistsError(
                f"File {csv_path} already exists, this should only happen on a uuid conflict and should be "
                "extremely rare"
            )
        self._log_file_path = str(csv_path)

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
    """Compute the number of windows for a sequence of the given length."""
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
    """Pre-compute window mappings for a split using a Parquet file.

    The Parquet file must contain ID and length columns as configured by
    ``SEQUENCE_ID_COLUMN_NAME`` and ``SEQUENCE_LENGTH_COLUMN_NAME``.

    Args:
        split_parquet_file: Path to a Parquet file with ID and length columns.
        output_window_db: Path to output window mapping database.
        window_size: Window size (default: 8192).
        stride: Stride between windows (default: 7992).
        window_min_length_threshold: Minimum length of windows to include (default: 0).
    """
    print(f"Creating window database at {output_window_db} from {split_parquet_file}")
    print(
        f"Using window_size={window_size}, stride={stride}, window_min_length_threshold={window_min_length_threshold}"
    )

    try:
        df = pol.read_parquet(split_parquet_file)
    except Exception as e:
        raise IOError(f"Failed to read Parquet file at {split_parquet_file}") from e

    if SEQUENCE_ID_COLUMN_NAME not in df.columns or SEQUENCE_LENGTH_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Parquet file {split_parquet_file} must contain '"
            f"{SEQUENCE_ID_COLUMN_NAME}' and '{SEQUENCE_LENGTH_COLUMN_NAME}' columns."
        )

    df = df.sort(SEQUENCE_ID_COLUMN_NAME)

    conn = sqlite3.connect(output_window_db)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=OFF;")
    cursor.execute("PRAGMA synchronous=OFF;")
    cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA cache_size=-1048576;")

    cursor.execute("DROP TABLE IF EXISTS window_mappings")
    cursor.execute("DROP TABLE IF EXISTS metadata")

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
            start_pos = i * stride if seq_len >= window_size else 0
            remaining = max(0, seq_len - start_pos)
            effective_window_len = min(window_size, remaining)

            if window_min_length_threshold and effective_window_len < window_min_length_threshold:
                skipped_windows += 1
                continue

            batch.append((global_window_idx, seq_id, i))
            global_window_idx += 1
            windows_added_for_seq += 1

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

    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("total_windows", int(global_window_idx)),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("distinct_sequences", int(total_sequences)),
    )

    conn.commit()
    conn.close()

    print(f"Finished. Found {total_sequences} sequences and {global_window_idx} total windows.")
    if window_min_length_threshold and skipped_windows > 0:
        print(f"Skipped {skipped_windows} windows due to window_min_length_threshold={window_min_length_threshold}.")


def main_precompute():
    """CLI entry point for pre-computing window databases."""
    parser = argparse.ArgumentParser(
        description="Pre-compute window mappings from a Parquet file into an SQLite database."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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
        help="If > 0, skip sequences shorter than this length when precomputing windows. Defaults to 0 (disabled).",
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
    main_precompute()
