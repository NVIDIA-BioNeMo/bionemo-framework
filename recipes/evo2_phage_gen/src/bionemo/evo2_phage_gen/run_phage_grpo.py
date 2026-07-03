# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Recipe-local NeMo-RL GRPO launcher for Evo2 phage optimization."""

from __future__ import annotations

import argparse
import os
import pprint

from omegaconf import OmegaConf


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Evo2 phage GRPO training")
    parser.add_argument("--config", type=str, default="configs/grpo_phage_megatron.yaml")
    return parser.parse_known_args()


def _select_grpo_trainer(dp_cfg: dict):
    if dp_cfg.get("enabled", False):
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync

        print("Running synchronous GRPO training (TransferQueue)")
        return grpo_train_sync
    from nemo_rl.algorithms.grpo import grpo_train

    print("Running synchronous GRPO training (legacy)")
    return grpo_train


def _register_recipe_extensions() -> None:
    """Register recipe-specific NeMo-RL processors and environments."""
    from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import register_env

    from bionemo.evo2_phage_gen.nemo_rl_processors import phage_prompt_data_processor

    processor_name = "phage_prompt_data_processor"
    if PROCESSOR_REGISTRY.get(processor_name) is phage_prompt_data_processor:
        pass
    elif processor_name in PROCESSOR_REGISTRY:
        raise ValueError(f"Dataset processor {processor_name} is already registered to a different function")
    else:
        register_processor("phage_prompt_data_processor", phage_prompt_data_processor)
    register_env("phage_qc", "bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment")
    ACTOR_ENVIRONMENT_REGISTRY["bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment"] = PY_EXECUTABLES.SYSTEM


def main() -> None:
    """Run GRPO with recipe-local Evo2 phage extensions."""
    os.environ.setdefault("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")
    try:
        from nemo_rl.algorithms.grpo import MasterConfig, async_grpo_train, setup
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.data.utils import setup_response_data
        from nemo_rl.distributed.virtual_cluster import init_ray
        from nemo_rl.models.generation import configure_generation_config
        from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
        from nemo_rl.utils.logger import get_next_experiment_dir
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NeMo-RL and its runtime dependencies are required for GRPO. "
            "Install the recipe environment, or repair an existing environment with "
            "evo2_phage_patch_nemo_rl --repair-install, before launching GRPO."
        ) from exc

    from bionemo.evo2_phage_gen.nemo_rl_patches import assert_nemo_rl_patch_runtime, patch_sha256

    print(f"Using NeMo-RL Evo2 patch SHA256: {patch_sha256()}")
    assert_nemo_rl_patch_runtime()
    _register_recipe_extensions()
    register_omegaconf_resolvers()
    args, overrides = _parse_args()
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    print("Applied CLI overrides")
    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, "A generation config is required for GRPO"
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(tokenizer, config.data, config.env)
    dp_cfg = config.data_plane or {}
    if dp_cfg.get("enabled", False):
        from nemo_rl.models.policy.tq_policy import TQPolicy

        def policy_factory(**kwargs):
            return TQPolicy(**kwargs, dp_cfg=dp_cfg)

    else:
        policy_factory = None

    (
        policy,
        policy_generation,
        _nemo_gym,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset, policy_factory=policy_factory)

    if "async_grpo" in config.grpo and config.grpo["async_grpo"]["enabled"]:
        async_config = config.grpo["async_grpo"]
        print("Running async GRPO training")
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        trainer = _select_grpo_trainer(dp_cfg)
        trainer(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
