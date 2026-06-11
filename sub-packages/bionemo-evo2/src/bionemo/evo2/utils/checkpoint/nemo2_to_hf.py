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

import argparse

from nemo.collections.llm.gpt.model.llama import HFLlamaExporter
from nemo.collections.llm.gpt.model.nemotron import HFNemotronExporter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, required=True, help="Model type to convert.", choices=["hyena", "mamba", "llama"]
    )
    parser.add_argument("--model-path", type=str, required=True, help="Model path to convert.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path for the converted model.")
    return parser.parse_args()


def main():
    """Convert a NeMo2 Evo2 model checkpoint to a Hugging Face model checkpoint."""
    args = parse_args()
    model_type = args.model_type
    if model_type == "hyena":
        raise ValueError("Hyena models are not supported for conversion to Hugging Face yet.")
    elif model_type == "mamba":
        exporter = HFNemotronExporter(args.model_path)
    elif model_type == "llama":
        exporter = HFLlamaExporter(args.model_path)
    else:
        raise ValueError(f"Invalid model type: {model_type}.")
    exporter.apply(args.output_dir)


if __name__ == "__main__":
    main()
