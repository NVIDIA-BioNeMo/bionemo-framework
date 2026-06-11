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
from pathlib import Path

from geneformer.export import export_hf_checkpoint


GENEFORMER_MODELS = [
    "Geneformer-V1-10M",
    "Geneformer-V2-104M",
    "Geneformer-V2-316M",
    "Geneformer-V2-104M_CLcancer",
]


def main():
    """Export the Geneformer models from Hugging Face to the Transformer Engine format."""
    parser = argparse.ArgumentParser(
        description="Convert Geneformer models from Hugging Face to Transformer Engine format"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=GENEFORMER_MODELS,
        help="Specific model to convert. If not provided, all models will be converted.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./checkpoint_export",
        help="Output directory path for the converted model. Defaults to './checkpoint_export'",
    )

    args = parser.parse_args()

    if args.model:
        if args.model not in GENEFORMER_MODELS:
            print(f"Error: '{args.model}' is not a valid model.\nAvailable models: {', '.join(GENEFORMER_MODELS)}")
            return

        print(f"Converting {args.model} from Hugging Face Hub...")
        export_hf_checkpoint(args.model, Path(args.output_path) / args.model)
    else:
        for model in GENEFORMER_MODELS:
            print(f"Converting {model} from Hugging Face Hub...")
            export_hf_checkpoint(model, Path(args.output_path) / model)


if __name__ == "__main__":
    main()
