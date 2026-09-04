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

import gc
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM

from amplify.export import export_hf_checkpoint


AMPLIFY_TAGS = ["AMPLIFY_120M", "AMPLIFY_350M"]


for tag in AMPLIFY_TAGS:
    print(f"Converting {tag}...")

    export_hf_checkpoint(tag, Path("./checkpoint_export"))

    gc.collect()
    torch.cuda.empty_cache()

    # Smoke test that the model can be loaded.
    model_te = AutoModelForMaskedLM.from_pretrained(
        f"./checkpoint_export/{tag}",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    del model_te
    gc.collect()
    torch.cuda.empty_cache()
