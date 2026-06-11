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


def small_training_cmd(
    path,
    max_steps,
    val_check,
    global_batch_size: int | None = None,
    devices: int = 1,
    additional_args: str = "",
):
    """Command for training."""
    cmd = (
        f"train_evo2 --mock-data --result-dir {path} --devices {devices} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* --limit-val-batches 1 "
        "--no-activation-checkpointing --add-bias-output --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} "
        f"--seq-length 16 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args} "
        f"{'--global-batch-size ' + str(global_batch_size) if global_batch_size is not None else ''}"
    )
    return cmd


def small_training_finetune_cmd(
    path,
    max_steps,
    val_check,
    prev_ckpt,
    devices: int = 1,
    global_batch_size: int | None = None,
    create_tflops_callback: bool = True,
    additional_args: str = "",
):
    """Command for finetuning."""
    cmd = (
        f"train_evo2 --mock-data --result-dir {path} --devices {devices} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* --limit-val-batches 1 "
        "--no-activation-checkpointing --add-bias-output --create-tensorboard-logger "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} "
        f"--seq-length 16 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args} --ckpt-dir {prev_ckpt} "
        f"{'--create-tflops-callback' if create_tflops_callback else ''} "
        f"{'--global-batch-size ' + str(global_batch_size) if global_batch_size is not None else ''}"
    )
    return cmd


def predict_cmd(ckpt_dir: str, output_dir: str, fasta_file_path: str, additional_args: str = ""):
    """Command fro predict."""
    cmd = (
        f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {ckpt_dir} --output-dir {output_dir} "
        "--model-size 1b_nv --num-layers 4 --hybrid-override-pattern SDH* --tensor-parallel-size 1 "
        f"--pipeline-model-parallel-size 1 --context-parallel-size 1 {additional_args}"
    )

    return cmd
