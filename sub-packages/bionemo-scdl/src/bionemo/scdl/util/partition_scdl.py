# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Partition a monolithic SCDL dataset into chunks."""

import shutil
from pathlib import Path

import numpy as np

from bionemo.scdl.schema.header import ChunkedInfo, SCDLHeader
from bionemo.scdl.util.scdl_constants import Backend, FileNames


def partition_scdl(
    input_path: Path,
    output_path: Path,
    chunk_size: int = 100_000,
    delete_original: bool = False,
) -> SCDLHeader:
    """Partition an SCDL dataset into chunks."""
    from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset

    input_path, output_path = Path(input_path), Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    output_path.mkdir(parents=True)

    # Load source dataset
    source_ds = SingleCellMemMapDataset(str(input_path))
    total_rows = len(source_ds)
    rowptr = source_ds.row_index
    if chunk_size <= 0:
        raise ValueError(f"Chunk size must be greater than 0, got {chunk_size}")
    if total_rows <= 0:
        raise ValueError(f"Total rows must be greater than 0, got {total_rows}")
    num_chunks = max(1, (total_rows + chunk_size - 1) // chunk_size)

    # Create chunks
    for chunk_id in range(num_chunks):
        row_start = chunk_id * chunk_size
        row_end = min(row_start + chunk_size, total_rows)
        chunk_dir = output_path / f"chunk_{chunk_id:05d}"
        chunk_dir.mkdir()

        data_start, data_end = int(rowptr[row_start]), int(rowptr[row_end])

        # Write chunk files using memmap slicing
        chunk_rowptr = rowptr[row_start : row_end + 1] - data_start
        with open(chunk_dir / FileNames.ROWPTR.value, "wb") as f:
            f.write(chunk_rowptr.astype(source_ds.dtypes[FileNames.ROWPTR.value]).tobytes())
        with open(chunk_dir / FileNames.DATA.value, "wb") as f:
            f.write(np.array(source_ds.data[data_start:data_end]).tobytes())
        with open(chunk_dir / FileNames.COLPTR.value, "wb") as f:
            f.write(np.array(source_ds.col_index[data_start:data_end]).tobytes())

    # Copy features and metadata
    for name in [FileNames.VAR_FEATURES.value, FileNames.OBS_FEATURES.value]:
        if (input_path / name).exists():
            shutil.copytree(input_path / name, output_path / name)
    for name in [FileNames.VERSION.value, FileNames.METADATA.value]:
        if (input_path / name).exists():
            shutil.copy(input_path / name, output_path / name)

    # Update header with chunked info
    header = source_ds.header if source_ds.header else SCDLHeader()
    header.backend = Backend.CHUNKED_MEMMAP_V0
    header.chunked_info = ChunkedInfo(chunk_size=chunk_size, num_chunks=num_chunks, total_rows=total_rows)
    header.save(str(output_path / FileNames.HEADER.value))

    if delete_original:
        del source_ds  # Release memmap handles
        shutil.rmtree(input_path)

    return header
