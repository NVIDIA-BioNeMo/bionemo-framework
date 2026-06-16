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


from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from bionemo.evo2.models.megatron.hyena.hyena_utils import hyena_no_weight_decay_cond


def hyena_no_weight_decay_cond_with_embeddings(name, param):
    """Condition for no weight decay for Hyena parameters with embeddings."""
    if "embedding" in name:
        return True
    return hyena_no_weight_decay_cond(name, param)


class Evo2TaxonomyLineage(BaseModel):
    """Pydantic model class that defines the source lineage of a DNA sequence."""

    domain: None | str = None
    phylum: None | str = None
    clazz: None | str = None
    order: None | str = None
    family: None | str = None
    genus: None | str = None
    species: None | str = None


class Evo2PreprocessingConfig(BaseModel):
    """Pydantic model class specifying the configuration schema for a preprocessed IndexedDataset (.bin, .idx)."""

    # Paths
    datapaths: list[Path] = []
    output_dir: None | Path = None
    output_prefix: None | str = None
    # Random Datasplit
    train_split: float = 0.7
    valid_split: float = 0.2
    test_split: float = 0.1
    # Overwrite existing binaries. Otherwise, skip already preprocessed datasets.
    overwrite: bool = False
    # Raw Preprocessing Transforms
    embed_reverse_complement: bool = False
    random_reverse_complement: float = 0.0
    random_lineage_dropout: float = 0.0
    transcribe: None | Literal["transcribe", "back_transcribe"] = None
    force_uppercase: bool = False
    indexed_dataset_dtype: str = "uint8"
    # Tokenization Transforms
    append_eod: bool = True
    enforce_sample_length: None | int = None
    ftfy: bool = False
    # NeMo Tokenizer Configuration
    hf_tokenizer_model_path: None | Path = None  # if left None, the 256-character nucleotide tokenizer will be used.
    hf_tokenizer_model_name: None | str = None
    hf_tokenizer_trust_remote_code: bool = (
        False  # Set to True for some custom local tokenizers. Not needed for the default.
    )
    # Compute Configuration
    # NOTE: If preprocessing a large amount of short individual sequences (< 1000 bp), do NOT use
    # multiprocessing (workers > 1) because sequence-level parallel IPC will dominate the preprocessing time!
    workers: int = 1
    preproc_concurrency: int = 100000
    chunksize: int = 1
    # Filters
    drop_empty_sequences: bool = False
    nnn_filter: bool = False
    # RNG
    seed: None | int = None
    # Evo2 Taxonomic Lineage Tags
    # SeqID Sub-String Indexing: "ABC" will have taxonomy data from "A".
    taxonomy_data: dict[str, Evo2TaxonomyLineage] = {}
    # Periodicity of injecting phylogenetic lineage tags in the sequence prior to tokenization.
    prompt_spacer_length: int = 131072
