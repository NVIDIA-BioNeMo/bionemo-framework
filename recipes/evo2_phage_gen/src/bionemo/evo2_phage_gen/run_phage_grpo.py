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
import pprint
from pathlib import Path

from omegaconf import OmegaConf


RECIPE_ROOT = Path(__file__).resolve().parents[3]
PAPER_RL_PROMPT_FILENAMES = {
    "phage_prompts_paper_useful_rl.jsonl",
    "phage_prompts_paper_useful_rl_validation_prompt10.jsonl",
}
VLLM_ACTOR_FQNS = (
    "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
    "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
    "nemo_rl.algorithms.async_utils.ReplayBuffer",
    "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
)
SYSTEM_ACTOR_FQNS = (
    "bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment",
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
)


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
    print("Materialized missing paper-useful RL prompt data:")
    for path in written_paths.values():
        print(f"  {path}")


def _validate_vllm_load_parity(config) -> dict[str, object] | None:
    """Bind the RL policy checkpoint to the standalone vLLM export before Ray starts."""
    generation = config.policy.get("generation") or {}
    if generation.get("backend") != "vllm":
        return None
    checkpoint = _config_path(config.checkpointing["pretrained_checkpoint"].get("path"))
    export = _config_path(config.policy.get("model_name"))
    tokenizer_config = config.policy.get("tokenizer") or {}
    tokenizer = _config_path(tokenizer_config.get("name"))
    if checkpoint is None or export is None or tokenizer is None:
        raise RuntimeError("vLLM RL requires explicit checkpoint, export, and tokenizer paths")

    from bionemo.evo2.vllm import load_parity

    evidence = load_parity.validate_rl_inference_load_parity(
        checkpoint=checkpoint,
        export=export,
        rl_tokenizer=tokenizer,
    )
    print(
        "Validated RL/standalone vLLM load parity: "
        f"manifest={evidence['export_manifest_sha256']}, "
        f"run_config={evidence['source_run_config_sha256']}, "
        f"tensors={evidence['tensor_count']}, "
        f"tokenizer={evidence['tokenizer_semantic_sha256']}"
    )
    return evidence


def _select_grpo_trainer(master_config, algorithm: str):
    dp_cfg = master_config.data_plane or {}
    if dp_cfg.get("enabled", False):
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync

        print(f"Running synchronous {algorithm.upper()} training (TransferQueue)")
        return grpo_train_sync
    from nemo_rl.algorithms.grpo import grpo_train

    print(f"Running synchronous {algorithm.upper()} training (legacy)")
    return grpo_train


def _unpack_grpo_setup_result(setup_result: tuple[object, ...]) -> tuple[object, ...]:
    """Validate the pinned NeMo-RL setup return contract before unpacking it."""
    if type(setup_result) is not tuple:
        raise TypeError(f"Pinned NeMo-RL setup must return a tuple, got {type(setup_result).__name__}")
    if len(setup_result) != 13:
        raise RuntimeError(f"Pinned NeMo-RL setup must return exactly 13 values, got {len(setup_result)}")
    return setup_result


def _register_recipe_extensions() -> None:
    """Register recipe-specific NeMo-RL processors and environments."""
    from nemo_rl.data.processors import PROCESSOR_REGISTRY, register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY, register_env

    from bionemo.evo2_phage_gen.nemo_rl_patches import vllm_actor_python_executable
    from bionemo.evo2_phage_gen.nemo_rl_processors import phage_prompt_data_processor

    processor_name = "phage_prompt_data_processor"
    if PROCESSOR_REGISTRY.get(processor_name) is phage_prompt_data_processor:
        pass
    elif processor_name in PROCESSOR_REGISTRY:
        raise ValueError(f"Dataset processor {processor_name} is already registered to a different function")
    else:
        register_processor(processor_name, phage_prompt_data_processor)
    env_name = "phage_qc"
    env_actor_fqn = "bionemo.evo2_phage_gen.nemo_rl_env.PhageQCEnvironment"
    registered_env = ENV_REGISTRY.get(env_name)
    if registered_env is None:
        register_env(env_name, env_actor_fqn)
    elif registered_env.get("actor_class_fqn") != env_actor_fqn:
        raise ValueError(f"Environment {env_name} is already registered to a different actor")
    actor_python = vllm_actor_python_executable()
    if not actor_python.is_file():
        raise RuntimeError(f"Pinned vLLM actor environment is missing: {actor_python}")
    for actor_fqn in SYSTEM_ACTOR_FQNS:
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.SYSTEM
    for actor_fqn in VLLM_ACTOR_FQNS:
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = str(actor_python)


def main(default_config: str = "configs/grpo_phage_megatron.yaml", default_algorithm: str = "config") -> None:
    """Run GRPO or GDPO with recipe-local Evo2 phage extensions."""
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
            "NeMo-RL and its runtime dependencies are required for GRPO/GDPO. "
            "Install the recipe environment, or repair an existing environment with "
            "evo2_phage_patch_nemo_rl --repair-install, before launching GRPO or GDPO."
        ) from exc

    from bionemo.evo2_phage_gen.nemo_rl_patches import assert_nemo_rl_patch_runtime, patch_sha256

    print(f"Using NeMo-RL Evo2 patch SHA256: {patch_sha256()}")
    assert_nemo_rl_patch_runtime()
    _register_recipe_extensions()
    register_omegaconf_resolvers()
    args, overrides = _parse_args(default_config, default_algorithm)
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config, algorithm = _apply_algorithm_override(config, args.algorithm)
    _ensure_prompt_data_files(config)
    print(f"Using RL algorithm frontend: {algorithm.upper()}")

    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    print("Applied CLI overrides")
    print("Final config:")
    pprint.pprint(config)
    _validate_vllm_load_parity(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    if config.policy["generation"] is None:
        raise RuntimeError("A generation config is required for GRPO/GDPO")
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
        teacher_worker_groups,
        alias_to_group_alias,
    ) = _unpack_grpo_setup_result(
        setup(config, tokenizer, dataset, val_dataset, policy_factory=policy_factory)
    )

    if "async_grpo" in config.grpo and config.grpo["async_grpo"]["enabled"]:
        async_config = config.grpo["async_grpo"]
        print(f"Running async {algorithm.upper()} training")
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
            teacher_worker_groups=teacher_worker_groups,
            alias_to_group_alias=alias_to_group_alias,
        )
    else:
        trainer = _select_grpo_trainer(master_config, algorithm)
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
