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


# %%
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


sys.path.append("/workspace/codonfm")
from src.tokenizer import Tokenizer

def main(pretraining_processed_data_dir: Path, data_dir: Path):
    tax_ids_to_remove = json.load(open(data_dir / Path("taxids_to_remove.json")))
    metadata = json.load(open(pretraining_processed_data_dir / "metadata.json"))
    tokenizer = Tokenizer()

    groups = set([x["file_name"][:-4] for x in metadata["file_metadata"]])  # noqa: C403
    counts = {g: np.zeros(tokenizer.vocab_size) for g in groups}
    for fm, cm in tqdm(zip(metadata["file_metadata"], metadata["chunks"]), total=len(metadata["file_metadata"])):
        group = fm["file_name"][:-4]
        if group in tax_ids_to_remove:
            curr_taxids_to_remove = set(tax_ids_to_remove[group])
        else:
            curr_taxids_to_remove = set()
        mmap = np.memmap(
            pretraining_processed_data_dir / cm["sequences"]["path"],
            dtype=cm["sequences"]["dtype"],
            mode="r",
            shape=tuple(cm["sequences"]["shape"]),
        )
        idx_mmap = np.memmap(
            pretraining_processed_data_dir / cm["index"]["path"], dtype=cm["index"]["dtype"], mode="r", shape=tuple(cm["index"]["shape"])
        )
        for start, end, taxid in idx_mmap:
            if taxid in curr_taxids_to_remove:
                continue
            seq = mmap[start:end]
            idx, count = np.unique(seq, return_counts=True)
            counts[group][idx] += count

    # %%
    for g in counts:
        counts[g] = counts[g].tolist()
    json.dump(counts, open(data_dir / "codon_counts_nopathogen.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check codon frequency")
    parser.add_argument("--pretraining_processed_data_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(Path(args.pretraining_processed_data_dir), Path(args.data_dir))