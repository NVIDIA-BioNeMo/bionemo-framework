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

"""Recipe-local NeMo-RL GRPO/GDPO launcher for Evo2 phage optimization."""

from __future__ import annotations

import argparse
import logging
import os
import pprint
from pathlib import Path

from omegaconf import OmegaConf


logger = logging.getLogger(__name__)

RECIPE_ROOT = Path(__file__).resolve().parents[3]
PAPER_RL_PROMPT_FILENAMES = {
    "phage_prompts_paper_useful_rl.jsonl",
    "phage_prompts_paper_useful_rl_validation_prompt10_96.jsonl",
}


def _parse_args(default_config: str, default_algorithm: str) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Evo2 phage GRPO or GDPO training")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument(
        "--algorithm",
        choices=("config", "grpo", "gdpo"),
        default=default_algorithm,
        help="Use config reward_output_mode, force scalar GRPO, or force positional multi-reward GDPO.",
    )
    return parser.parse_known_args()


def _apply_algorithm_override(config, algorithm: str) -> tuple[object, str]:
    """Apply launcher-level GRPO/GDPO reward mode overrides."""
    if algorithm == "config":
        reward_mode = str(OmegaConf.select(config, "env.phage_qc.reward_output_mode", default="scalar")).lower()
        resolved_algorithm = "gdpo" if reward_mode == "gdpo" else "grpo"
    else:
        resolved_algorithm = algorithm
        reward_mode = "gdpo" if algorithm == "gdpo" else "scalar"
        OmegaConf.update(config, "env.phage_qc.reward_output_mode", reward_mode, merge=True)

    OmegaConf.update(config, "grpo.adv_estimator.name", resolved_algorithm, merge=True)
    if OmegaConf.select(config, "grpo.adv_estimator.normalize_rewards", default=None) is None:
        OmegaConf.update(config, "grpo.adv_estimator.normalize_rewards", "${grpo.normalize_rewards}", merge=True)
    if OmegaConf.select(config, "grpo.adv_estimator.use_leave_one_out_baseline", default=None) is None:
        OmegaConf.update(
            config,
            "grpo.adv_estimator.use_leave_one_out_baseline",
            "${grpo.use_leave_one_out_baseline}",
            merge=True,
        )
    return config, resolved_algorithm


def _config_path(path_like: str | None) -> Path | None:
    """Resolve recipe-relative config data paths while preserving absolute paths."""
    if not path_like:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else RECIPE_ROOT / path


def _ensure_prompt_data_files(config) -> None:
    """Materialize deterministic recipe-owned prompt data referenced by configs."""
    configured_paths = [
        _config_path(OmegaConf.select(config, "data.train.data_path")),
        _config_path(OmegaConf.select(config, "data.validation.data_path")),
    ]
    prompt_paths = [path for path in configured_paths if path is not None and path.name in PAPER_RL_PROMPT_FILENAMES]
    if not prompt_paths:
        return

    missing_paths = [path for path in prompt_paths if not path.exists()]
    if not missing_paths:
        return

    from bionemo.evo2_phage_gen.generation import ensure_paper_useful_rl_prompt_files

    data_dir = missing_paths[0].parent
    written_paths = ensure_paper_useful_rl_prompt_files(data_dir)
    logger.info("Materialized missing paper-useful RL prompt data:")
    for path in written_paths.values():
        logger.info("  %s", path)


def _select_grpo_trainer(*, data_plane_enabled: bool, algorithm: str):
    if data_plane_enabled:
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync

        logger.info("Running synchronous %s training (TransferQueue)", algorithm.upper())
        return grpo_train_sync
    from nemo_rl.algorithms.grpo import grpo_train

    logger.info("Running synchronous %s training (legacy)", algorithm.upper())
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
        register_processor(processor_name, phage_prompt_data_processor)
    register_env("phage_qc", "bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment")
    ACTOR_ENVIRONMENT_REGISTRY["bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment"] = PY_EXECUTABLES.SYSTEM


def _init_ray(upstream_init_ray, *, include_dashboard: bool, num_cpus: int | None = None) -> None:
    """Initialize Ray through the recipe-supported NeMo-RL interface."""
    options = {"include_dashboard": include_dashboard}
    if num_cpus is not None:
        options["num_cpus"] = num_cpus
    upstream_init_ray(**options)


def main(default_config: str = "configs/grpo_phage_megatron.yaml", default_algorithm: str = "config") -> None:
    """Run GRPO or GDPO with recipe-local Evo2 phage extensions."""
    os.environ.setdefault("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        from nemo_rl.algorithms.grpo import MasterConfig, async_grpo_train, setup
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.data.utils import setup_response_data
        from nemo_rl.distributed.virtual_cluster import init_ray as upstream_init_ray
        from nemo_rl.models.generation import configure_generation_config
        from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
        from nemo_rl.utils.logger import get_next_experiment_dir
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NeMo-RL and its runtime dependencies are required for GRPO/GDPO. "
            "Install the recipe environment, or repair an existing environment with "
            "evo2_phage_setup_nemo_rl, before launching GRPO or GDPO."
        ) from exc

    from bionemo.evo2_phage_gen.nemo_rl_setup import assert_nemo_rl_runtime

    assert_nemo_rl_runtime()
    _register_recipe_extensions()
    register_omegaconf_resolvers()
    args, overrides = _parse_args(default_config, default_algorithm)
    config = load_config(args.config)
    logger.info("Loaded configuration from: %s", args.config)
    if overrides:
        logger.info("Overrides: %s", overrides)
        config = parse_hydra_overrides(config, overrides)
    config, algorithm = _apply_algorithm_override(config, args.algorithm)
    _ensure_prompt_data_files(config)
    logger.info("Using RL algorithm frontend: %s", algorithm.upper())

    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    logger.info("Applied CLI overrides")
    logger.info("Final config:\n%s", pprint.pformat(config))

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    logger.info("Using log directory: %s", config.logger["log_dir"])
    if config.checkpointing["enabled"]:
        logger.info("Using checkpoint directory: %s", config.checkpointing["checkpoint_dir"])

    include_ray_dashboard = os.environ.get("NEMO_RL_RAY_DASHBOARD", "0").lower() in {"1", "true", "yes"}
    ray_num_cpus = int(os.environ["NEMO_RL_RAY_NUM_CPUS"]) if os.environ.get("NEMO_RL_RAY_NUM_CPUS") else None
    _init_ray(upstream_init_ray, include_dashboard=include_ray_dashboard, num_cpus=ray_num_cpus)
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, "A generation config is required for GRPO/GDPO"
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
    data_plane_enabled = bool(dp_cfg.get("enabled", False))
    if data_plane_enabled:
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
        experiment_logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset, policy_factory=policy_factory)

    if "async_grpo" in config.grpo and config.grpo["async_grpo"]["enabled"]:
        async_config = config.grpo["async_grpo"]
        logger.info("Running async %s training", algorithm.upper())
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=experiment_logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        trainer = _select_grpo_trainer(data_plane_enabled=data_plane_enabled, algorithm=algorithm)
        trainer(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            experiment_logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
