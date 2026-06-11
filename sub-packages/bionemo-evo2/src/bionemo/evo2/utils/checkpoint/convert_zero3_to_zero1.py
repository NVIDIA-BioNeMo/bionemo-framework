# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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


#!/usr/bin/env python

import argparse
import os
import time
from multiprocessing import Pool
from typing import List, Optional

import zero3_conversion_lib
from zero3_conversion_lib import get_elapsed, process_single_rank


def convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir: str,
    output_dir: str,
    tag: Optional[str] = None,
    exclude_frozen_parameters: bool = False,
    mp_size: int = 8,
    overwrite: bool = False,
    num_workers: int = 1,
    ranks_to_process: Optional[List[int]] = None,
):
    """Converts a DeepSpeed Zero-3 checkpoint to a PyTorch FP32 state_dict.

    Args:
        checkpoint_dir (str): Path to the desired checkpoint folder.
        output_dir (str): Directory to save the PyTorch FP32 state_dict output files.
        tag (Optional[str]): Checkpoint tag used as a unique identifier or sub-directory that contains the checkpoint.
        exclude_frozen_parameters (bool): Whether to exclude frozen parameters.
        mp_size (int): Model parallel size of the source checkpoint.
        overwrite (bool): Whether to overwrite existing MP shards.
        num_workers (int): Number of workers to use for processing.
        ranks_to_process (Optional[List[int]]): List of ranks to process.

    Raises:
        FileNotFoundError: If the checkpoint directory does not exist.
    """
    ds_checkpoint_dir = os.path.join(checkpoint_dir, tag) if tag is not None else checkpoint_dir

    if not os.path.isdir(ds_checkpoint_dir):
        raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

    output_dir = os.path.join(output_dir, tag) if tag is not None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    num_workers = min(num_workers, mp_size)

    if ranks_to_process is not None:
        ranks_to_process = list(ranks_to_process)
        assert len(ranks_to_process) <= mp_size, f"Expected {mp_size} ranks to process, got {len(ranks_to_process)}"
        assert all(0 <= r < mp_size for r in ranks_to_process), (
            f"Expected ranks to be in range [0, {mp_size}), got {ranks_to_process}"
        )
    else:
        ranks_to_process = list(range(mp_size))

    print(f"Processing ranks: {ranks_to_process}", flush=True)

    start = time.time()
    if num_workers > 1:
        with Pool(num_workers) as p:
            p.starmap(
                process_single_rank,
                [(i, ds_checkpoint_dir, output_dir, overwrite, exclude_frozen_parameters) for i in ranks_to_process],
            )
    else:
        for i in ranks_to_process:
            process_single_rank(i, ds_checkpoint_dir, output_dir, overwrite, exclude_frozen_parameters)

    total_time = get_elapsed(time.time() - start)
    print(f"All done!\n-> Total time: {total_time}\n-> All outputs written to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir", type=str, help="path to the desired checkpoint folder, e.g., path/checkpoint-12"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="directory to the pytorch fp32 state_dict output files(e.g. path/checkpoint-12-output/)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP shards")
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default=None,
        help="Checkpoint tag used as a unique identifier or sub-directory that contains the checkpoint, e.g. 'global_step1' or 'latest'.",
    )
    parser.add_argument("--exclude_frozen_parameters", action="store_true", help="exclude frozen parameters")
    parser.add_argument("-d", "--debug", action="store_true", help="enable debug")
    parser.add_argument("--mp_size", required=True, type=int, help="Model parallel size of source checkpoint")
    parser.add_argument("--rank_start", default=None, type=int, help="Start rank to process")
    parser.add_argument("--rank_end", default=None, type=int, help="End rank to process")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers to use for processing")
    args = parser.parse_args()

    if args.rank_start is not None:
        if args.rank_end is None:
            args.rank_end = args.mp_size - 1
        else:
            assert args.rank_end < args.mp_size, "Expected end rank to be less than mp_size"

        assert args.rank_start < args.rank_end, "Expected start rank to be less than end rank"
        assert args.rank_start >= 0, "Expected start rank to be greater than 0"
        args.ranks_to_process = list(range(args.rank_start, args.rank_end + 1))
    else:
        args.ranks_to_process = list(range(args.mp_size))

    print("Args:")
    for k, v in args.__dict__.items():
        print(f"  {k}: {v}", flush=True)
    print("")
    zero3_conversion_lib.debug = args.debug

    convert_zero_checkpoint_to_fp32_state_dict(
        args.checkpoint_dir,
        args.output_dir,
        tag=args.tag,
        exclude_frozen_parameters=args.exclude_frozen_parameters,
        mp_size=args.mp_size,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        ranks_to_process=args.ranks_to_process,
    )
