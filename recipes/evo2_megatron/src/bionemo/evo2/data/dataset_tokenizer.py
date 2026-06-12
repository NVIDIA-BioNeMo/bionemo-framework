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

import ftfy
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer, build_tokenizer

from bionemo.evo2.utils.config import Evo2PreprocessingConfig


REPO_BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
DEFAULT_HF_TOKENIZER_MODEL_PATH = str(REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_256")
DEFAULT_HF_TOKENIZER_MODEL_PATH_512 = str(REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_512")


class Evo2DatasetTokenizer:
    """Dataset Tokenizer for Evo2."""

    def __init__(self, params: Evo2PreprocessingConfig | None = None):
        """Initialize the Evo2Tokenizer."""
        # Pass all NeMo2/Megatron-compliant parameters associated with config.Evo2PreprocessingConfig.
        self.params: Evo2PreprocessingConfig = params if params is not None else Evo2PreprocessingConfig()
        if self.params.hf_tokenizer_model_path is not None:
            hf_tokenizer_model_or_path = str(self.params.hf_tokenizer_model_path)
            hf_tokenizer_desc: str = Path(hf_tokenizer_model_or_path).name
            assert Path(hf_tokenizer_model_or_path).exists(), (
                f"Hugging Face tokenizer model path {hf_tokenizer_model_or_path} does not exist."
            )
        elif self.params.hf_tokenizer_model_name is not None:
            hf_tokenizer_model_or_path = str(self.params.hf_tokenizer_model_name)
            hf_tokenizer_desc = hf_tokenizer_model_or_path.replace("/", "--").replace(":", "--")
        else:
            hf_tokenizer_model_or_path = DEFAULT_HF_TOKENIZER_MODEL_PATH
            hf_tokenizer_desc = Path(hf_tokenizer_model_or_path).name
            assert Path(hf_tokenizer_model_or_path).exists(), (
                f"Default Hugging Face tokenizer model path {hf_tokenizer_model_or_path} does not exist."
            )
        self.hf_tokenizer_desc = hf_tokenizer_desc
        self.tokenizer: MegatronTokenizer = build_tokenizer(
            TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                hf_tokenizer_kwargs={"trust_remote_code": self.params.hf_tokenizer_trust_remote_code},
                tokenizer_model=hf_tokenizer_model_or_path,
            )
        )

    def tokenize(
        self,
        text: str | list[str],
        use_ftfy: bool = False,
        enforce_sample_length: None | int = None,
        append_eod: bool = False,
        drop_empty_sequences: bool = False,
    ):
        """Tokenize the input text data for Evo2."""
        if isinstance(text, str):
            text = [text]
        # Tokenize a document or batch of strings.
        doc_ids = []
        for i, t in enumerate(text):
            if use_ftfy:
                t_fixed = ftfy.fix_text(t)
            else:
                t_fixed = t
            # Tokenize the string.
            if hasattr(self.tokenizer, "text_to_ids"):
                # Handle the legacy NeMo2 style tokenizer.
                text_ids: list = self.tokenizer.text_to_ids(t_fixed)
            else:
                # Handle the new Megatron-Bridge style tokenizer.
                text_ids: list = self.tokenizer.tokenize(t_fixed)
            if drop_empty_sequences and len(text_ids) == 0:
                continue
            # Append EOD token (EOD ID: 0) if appropriate.
            eod_length = int(append_eod and i == len(text) - 1)
            token_length = len(text_ids) + eod_length
            text_ids += [0] * eod_length
            if enforce_sample_length is not None:
                # Pad shorter sequences (Pad ID: 1) and except excessive sequences.
                if token_length > enforce_sample_length:
                    raise ValueError(
                        "Detected input text with a length greater than the maximum "
                        f"possible sample length of {enforce_sample_length}.)"
                    )
                else:
                    text_ids += [1] * (enforce_sample_length - token_length)
            # Append to document.
            doc_ids.append(text_ids)
        return doc_ids
