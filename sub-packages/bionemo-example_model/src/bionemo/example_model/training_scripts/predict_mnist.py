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

from nemo import lightning as nl

from bionemo.core import BIONEMO_CACHE_DIR
from bionemo.example_model.lightning.lightning_basic import (
    BionemoLightningModule,
    ExampleFineTuneConfig,
    MNISTDataModule,
)


def run_predict(finetune_dir: str, test_length: int):
    """Run the prediction step.

    Args:
        finetune_dir: The directory with the previous step
        test_length: The length of the test step.

    Returns:
        tensor: the outputs of the model.
    """
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        always_save_context=True,
    )

    test_run_trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    lightning_module3 = BionemoLightningModule(config=ExampleFineTuneConfig(initial_ckpt_path=finetune_dir))
    new_data_module = MNISTDataModule(data_dir=str(BIONEMO_CACHE_DIR), batch_size=test_length, output_log=False)

    results = test_run_trainer.predict(lightning_module3, datamodule=new_data_module)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_dir", type=str, help="The directory with the fine-tuned model. ")
    args = parser.parse_args()
    test_length = 10_000
    results = run_predict(args.finetune_dir, test_length)
    print(results)
