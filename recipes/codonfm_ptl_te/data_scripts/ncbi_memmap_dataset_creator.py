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


import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm


sys.path.append("/workspace/codonfm")
from src.tokenizer import Tokenizer


def tokenize_sequence(args):  # noqa: D103
    cds_sequence, tokenizer = args
    return "".join(map(chr, tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cds_sequence))))


def preload_csv_files(  # noqa: D103
    data_path,
    tokenizer,
    save_path,
    chunk_size,
    num_workers=cpu_count(),
    min_cds_len=100,
    max_cds_len=1.5e5,
    batch_size=100000,
    parquet_compression=None,
):
    chunks_metadata_path = os.path.join(save_path, "chunks_metadata.json")

    chunks_metadata = []
    processed_metadata = []
    current_chunk_tokens = 0
    chunk_counter = 0
    completed_chunk_ids = set()

    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    processed_data_path = os.path.join(save_path, "data_processed")
    os.makedirs(processed_data_path, exist_ok=True)

    files_list = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])
    # Keep a persistent worker pool during tokenization to reduce spawn overhead
    with Pool(processes=num_workers) as pool:
        for file_name in tqdm(files_list, desc="Files", unit="file"):
            file_path = os.path.join(data_path, file_name)
            cached_file_path = os.path.join(processed_data_path, f"{file_name}.parquet")
            if os.path.exists(cached_file_path):
                print(f"Found cached tokens for {file_name} -> {cached_file_path}")
            else:
                print(f"Processing and tokenizing {file_name} (streamed batches)...")
                # Read only required columns to reduce memory
                df_all = pl.read_csv(file_path, columns=["cds", "taxid"])
                bs = int(batch_size)
                writer = None
                try:
                    rows_total = df_all.shape[0]
                    pbar_rows = tqdm(total=rows_total, desc=f"Rows {file_name}", unit="row", leave=False)
                    for bi in range(0, rows_total, bs):
                        cds_sequences = df_all[bi : bi + bs, "cds"].to_list()
                        taxid_batch = df_all[bi : bi + bs, "taxid"].to_list()
                        # Tokenize this batch only using persistent pool
                        tokenized_sequences = list(
                            tqdm(
                                pool.imap(
                                    tokenize_sequence,
                                    [(seq, tokenizer) for seq in cds_sequences],
                                    chunksize=max(2000, bs // 10),
                                ),
                                total=len(cds_sequences),
                                desc=f"Tokenizing {file_name} [{bi}:{bi + len(cds_sequences)}]",
                                leave=False,
                            )
                        )
                        df_batch = pl.DataFrame(
                            {
                                "taxid": taxid_batch,
                                "cds_tokens": tokenized_sequences,
                            }
                        ).with_columns(
                            pl.col("cds_tokens").str.len_chars().alias("cds_tokens_length"),
                            pl.col("cds_tokens")
                            .map_elements(lambda s: s.encode("utf-8"), return_dtype=pl.Binary)
                            .alias("cds_tokens_bytes"),
                        )
                        # Stream append to parquet via Arrow writer
                        tbl = df_batch.to_arrow()
                        if writer is None:
                            writer = pq.ParquetWriter(cached_file_path, tbl.schema, compression=parquet_compression)
                        writer.write_table(tbl)
                        del df_batch, tbl, tokenized_sequences, cds_sequences, taxid_batch
                        pbar_rows.update(min(bs, rows_total - bi))
                    pbar_rows.close()
                finally:
                    if writer is not None:
                        writer.close()
                del df_all
                print(f"Saved cached tokens to {cached_file_path}")
            # Load minimal columns needed for planning chunks (lengths/taxids only) for this file
            df_meta = pl.read_parquet(cached_file_path, columns=["taxid", "cds_tokens_length"])
            taxid_series = df_meta["taxid"].to_numpy()
            token_lengths = df_meta["cds_tokens_length"].to_numpy()
            del df_meta
            remaining_chunks_metadata = []
            print("Computing new chunks metadata for", file_name, "...")

            start_idx = 0

            for idx, seq_len in enumerate(tqdm(token_lengths, leave=False)):
                if chunk_size and current_chunk_tokens + seq_len > chunk_size and current_chunk_tokens > 0:
                    remaining_chunks_metadata.append([chunk_counter, (file_path, start_idx, idx - 1)])
                    chunk_counter += 1

                    start_idx = idx
                    current_chunk_tokens = 0
                current_chunk_tokens += seq_len

            if start_idx <= len(token_lengths) - 1:
                remaining_chunks_metadata.append([chunk_counter, (file_path, start_idx, len(token_lengths) - 1)])
                chunk_counter += 1
                current_chunk_tokens = 0
            chunks_metadata += remaining_chunks_metadata

            # Always write chunks in-process to avoid duplicating large arrays across workers
            for chunk_id, (file_path, start_row_idx, end_row_idx) in tqdm(
                remaining_chunks_metadata, total=len(remaining_chunks_metadata), leave=False
            ):
                result = save_chunk(
                    chunk_id,
                    (file_path, start_row_idx, end_row_idx),
                    save_path,
                    token_lengths,
                    taxid_series,
                    cached_file_path,
                )
                processed_metadata.append(result)
                completed_chunk_ids.add(chunk_id)

            # Update chunks metadata file incrementally (optional but useful for resume)
            with open(chunks_metadata_path, "w") as f:
                json.dump(chunks_metadata, f)

    final_metadata = {
        "chunks": processed_metadata,
        "tokenizer": tokenizer.get_vocab(),
        "file_metadata": [
            {"file_name": os.path.basename(chunk[1][0]), "start": chunk[1][1], "end": chunk[1][2]}
            for chunk in chunks_metadata
        ],
    }

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(final_metadata, f, indent=4)
    return


def save_chunk(chunk_id, chunk_info, save_path, token_lengths, taxid_series, cached_file_path):  # noqa: D103
    sequence_mmap_path = os.path.join(save_path, f"sequences_chunk{chunk_id}.mmap")
    index_mmap_path = os.path.join(save_path, f"index_chunk{chunk_id}.mmap")

    _, start_row_idx, end_row_idx = chunk_info

    token_lengths_slice = token_lengths[start_row_idx : end_row_idx + 1]
    taxid_slice = taxid_series[start_row_idx : end_row_idx + 1]

    # Reconstruct the token bytes for this slice only to avoid a giant global buffer
    # Use positional slice to avoid deprecated streaming engine and row counting
    rows_len = int(end_row_idx - start_row_idx + 1)
    lf = pl.scan_parquet(cached_file_path).select(["cds_tokens_bytes"]).slice(start_row_idx, rows_len)
    df_tokens = lf.collect()
    # Join encoded bytes in-order for the chunk (already bytes)
    sequences_bytes = b"".join(df_tokens["cds_tokens_bytes"].to_list())
    sequences_flat_slice = np.frombuffer(sequences_bytes, dtype=np.uint8)

    end_idxs_global = np.cumsum(token_lengths_slice)
    start_idxs_global = np.concatenate(([0], end_idxs_global[:-1]))

    indices_array = np.column_stack((start_idxs_global, end_idxs_global, taxid_slice)).astype(np.uint32)

    sequence_mmap = np.memmap(sequence_mmap_path, dtype="uint8", mode="w+", shape=sequences_flat_slice.shape)
    index_mmap = np.memmap(index_mmap_path, dtype="uint32", mode="w+", shape=indices_array.shape)

    sequence_mmap[:] = sequences_flat_slice[:]
    index_mmap[:] = indices_array[:]

    sequence_mmap.flush()
    index_mmap.flush()

    return {
        "sequences": {"path": os.path.basename(sequence_mmap_path), "shape": sequence_mmap.shape, "dtype": "uint8"},
        "index": {"path": os.path.basename(index_mmap_path), "shape": index_mmap.shape, "dtype": "uint32"},
    }


def update_completed_chunks(save_path, completed_chunk_ids):  # noqa: D103
    completed_chunks_file = os.path.join(save_path, "completed_chunks.json")
    with open(completed_chunks_file, "w") as f:
        json.dump(completed_chunk_ids, f)


def load_completed_chunks(save_path):  # noqa: D103
    completed_chunks_file = os.path.join(save_path, "completed_chunks.json")
    if os.path.exists(completed_chunks_file):
        with open(completed_chunks_file) as f:
            return set(json.load(f))
    return set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Chunk-level parallel processing.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000_000)
    parser.add_argument("--min-cds-len", type=int, default=100)
    parser.add_argument("--max-cds-len", type=float, default=1.5e5)
    parser.add_argument("--num-workers", type=int, default=cpu_count())

    args = parser.parse_args()

    tokenizer = Tokenizer(
        cls_token="<CLS>",
        bos_token="<CLS>",
        sep_token="<SEP>",
        unk_token="<UNK>",
        pad_token="<PAD>",
        mask_token="<MASK>",
        padding_side="right",
        truncation="right",
        seq_type="dna",
    )

    print("Preloading CSV files...")
    preloaded_data_tokenized = preload_csv_files(
        args.data_path,
        tokenizer,
        args.save_path,
        args.chunk_size,
        num_workers=args.num_workers,
        min_cds_len=args.min_cds_len,
        max_cds_len=args.max_cds_len,
    )

    print(f"✅ Processing complete. Outputs saved to {args.save_path}")
