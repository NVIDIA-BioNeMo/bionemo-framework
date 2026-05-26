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


import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Type

import torch
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.config import (
    ConfigContainer,
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.bridge.training.gpt_step import get_batch_from_iterator
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.optimizer import (
    ParamGroupOverride,
    ParamKey,
    ParamPredicate,
)
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from bionemo.evo2.models.megatron.hyena.hyena_config import HyenaConfig as _HyenaConfigForFlops
from bionemo.evo2.models.megatron.hyena.hyena_layer_specs import get_hyena_stack_spec
from bionemo.evo2.models.megatron.hyena.hyena_model import HyenaModel as MCoreHyenaModel
from bionemo.evo2.models.megatron.hyena.hyena_utils import hyena_no_weight_decay_cond


def _patch_megatron_dataset_helper_compile() -> None:
    """Skip Megatron's runtime helper build when a wheel already ships the extension."""
    from megatron.core.datasets import utils as dataset_utils

    original_compile_helpers = dataset_utils.compile_helpers
    if getattr(original_compile_helpers, "_evo2_prebuilt_helper_guard", False):
        guarded_compile_helpers = original_compile_helpers
    else:

        def guarded_compile_helpers() -> None:
            datasets_dir = Path(dataset_utils.__file__).resolve().parent
            if not (datasets_dir / "Makefile").exists() and list(datasets_dir.glob("helpers_cpp*.so")):
                return None
            return original_compile_helpers()

        guarded_compile_helpers._evo2_prebuilt_helper_guard = True
        dataset_utils.compile_helpers = guarded_compile_helpers

    bridge_initialize = sys.modules.get("megatron.bridge.training.initialize")
    if bridge_initialize is not None:
        bridge_initialize.compile_helpers = guarded_compile_helpers


_patch_megatron_dataset_helper_compile()
register_allowed_target_prefix("bionemo.evo2.")


def get_vocab_size(*args, **kwargs):
    raise NotImplementedError("FIXME get_vocab_size is not implemented Find it in megatron bridge")


def gpt_data_step(*args, **kwargs):
    raise NotImplementedError("FIXME gpt_data_step is not implemented Find it in megatron bridge")


@dataclass
class HyenaOptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Hyena-specific optimizer config override provider."""

    no_weight_decay_embeddings: bool = False

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        """Build config overrides for weight decay based on scheduler configuration.

        This function creates parameter-specific overrides for weight decay behavior.
        By default, weight decay is skipped for bias parameters and 1D parameters.
        For Qwen3-Next models, weight decay is applied to q_layernorm and k_layernorm.
        """
        optimizer_config = context.optimizer_config
        config_overrides: dict[ParamKey, ParamGroupOverride] = {}
        param_length_1_match = ParamPredicate(name="param_len_1", fn=lambda param: len(param.shape) == 1)
        name_tuple: tuple[str, ...] = (
            "*.bias",
            "*.filter.p",
            "*.filter.R",
            "*.filter.gamma",
            "*.short_conv.short_conv_weight",
        )
        if self.no_weight_decay_embeddings:
            name_tuple += ("*embedding*",)
        param_wd_mult_key = ParamKey(
            name=name_tuple,  # type: ignore
            predicate=param_length_1_match,
        )

        config_overrides[param_wd_mult_key] = ParamGroupOverride(wd_mult=0.0)  # type: ignore

        if optimizer_config.decoupled_lr is not None:
            decoupled_lr_config: ParamGroupOverride = {"max_lr": optimizer_config.decoupled_lr}
            decoupled_param_key = ParamKey(attr="is_embedding_or_output_parameter")
            if optimizer_config.decoupled_min_lr is not None:
                decoupled_lr_config["min_lr"] = optimizer_config.decoupled_min_lr
            config_overrides[decoupled_param_key] = decoupled_lr_config
        return config_overrides


class HyenaInferenceContext(StaticInferenceContext):
    """Hyena-specific inference context."""

    def reset(self):
        """Reset the inference context."""
        super().reset()  # standard state reset for GPT models
        for key in dir(self):
            # Remove all of the state that we add in hyena.py
            if "filter_state_dict" in key:
                delattr(self, key)


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled
        pg_collection: Process group collection
    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, and max_seqlen
    """
    # Determine pipeline stage role via process group collection
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)
    if (not is_first) and (not is_last):
        return None, None, None, None, None, None, None, None
    need_attention_mask = not getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True)
    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        need_attention_mask,
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    attention_mask = batch.get("attention_mask")
    if need_attention_mask and attention_mask is None:
        raise ValueError("Attention mask is required but not found in the batch")

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        attention_mask,
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
    )


def _forward_step_common(
    state: GlobalState, data_iterator: Iterable, model: MCoreHyenaModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and loss mask
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        tokens, labels, loss_mask, _, position_ids, cu_seqlens, cu_seqlens_argmin, max_seqlen = get_batch(
            data_iterator, state.cfg, use_mtp, pg_collection=pg_collection
        )
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": None,
        "loss_mask": loss_mask,
        "labels": labels,
    }

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
        }
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(tokens, position_ids, None, labels=labels, loss_mask=loss_mask)
            return schedule_plan, loss_mask
        else:
            output_tensor = model(**forward_args)

    return output_tensor, loss_mask


def hyena_forward_step(
    state: GlobalState, data_iterator: Iterable, model: MCoreHyenaModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = _create_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )

    return output, loss_function


def _create_loss_function(loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool) -> partial:
    """Create a partial loss function with the specified configuration.

    Args:
        loss_mask: Used to mask out some portions of the loss
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        A partial function that can be called with output_tensor to compute the loss
    """
    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


@dataclass
class HyenaModelProvider(TransformerConfig, ModelProviderMixin[MCoreHyenaModel]):
    """Configuration dataclass for Hyena.

    For adjusting ROPE when doing context extension, set seq_len_interpolation_factor relative to 8192.
    For example, if your context length is 512k, then set the factor to 512k / 8k = 64.
    """

    # From megatron.core.models.hyena.hyena_model.HyenaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = 2
    hidden_size: int = 1024
    num_attention_heads: int = 8
    num_groups_hyena: int = None
    num_groups_hyena_medium: int = None
    num_groups_hyena_short: int = None
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 2048
    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "rope"
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = True
    fp32_residual_connection: bool = False
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    attention_backend: AttnBackend = AttnBackend.flash
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 4
    forward_step_fn: Callable = hyena_forward_step
    data_step_fn: Callable = gpt_data_step  # FIXME do megatron bridge thing instead of this
    tokenizer_model_path: str = None
    hyena_init_method: str = None
    hyena_output_layer_init_method: str = None
    hyena_filter_no_wd: bool = True
    remove_activation_post_first_layer: bool = True
    add_attn_proj_bias: bool = True
    cross_entropy_loss_fusion: bool = False  # Faster but lets default to False for more precision
    tp_comm_overlap: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    add_bias_output: bool = False
    use_te: bool = True
    to_upper: str = "normalized_weighted"  # choose between "weighted" and "normalized_weighted"
    use_short_conv_bias: bool = False
    # Use this if you want to turn FP8 on for the linear layer in the mixer only. When using this, do not set
    #  Fp8 in the mixed precision plugin.
    vortex_style_fp8: bool = False
    use_subquadratic_ops: bool = False
    share_embeddings_and_output_weights: bool = True
    unfused_rmsnorm: bool = False  # Use unfused RMSNorm + TELinear for dense projection
    plain_row_linear: bool = False  # Use plain pytorch implementation instead of Megatron's row parallel linears
    vocab_size: Optional[int] = None
    should_pad_vocab: bool = False

    def __post_init__(self):
        """Post-initialization hook that sets up weight decay conditions."""
        super().__post_init__()
        self.hyena_no_weight_decay_cond_fn = hyena_no_weight_decay_cond if self.hyena_filter_no_wd else None

    def _get_num_floating_point_operations(self, batch_size: int) -> int:
        """Get the number of floating point operations for the model. This overrides the default in megatron bridge."""
        # Ported from https://github.com/NVIDIA-NeMo/NeMo/blob/45a3b5cad3434692b1fb805934913d95be8668ea/nemo/utils/hyena_flops_formulas.py
        """Model FLOPs for Hyena family. FPL = 'flops per layer'."""

        # TODO(@cye): For now, pull the Hyena defaults directly from a constant dataclass. Merge this config with the NeMo
        #   model config.
        hyena_config = _HyenaConfigForFlops()
        # Hyena Parameters
        hyena_short_conv_L = hyena_config.short_conv_L  # noqa: N806
        hyena_short_conv_len = hyena_config.hyena_short_conv_len
        hyena_medium_conv_len = hyena_config.hyena_medium_conv_len

        def _hyena_layer_count(model_pattern: Optional[str]):
            """Count how many small, medium, and large Hyena layers there are in the model. Also, count the number of Attention layers."""
            S, D, H, A = 0, 0, 0, 0  # noqa: N806
            if model_pattern is None:
                return 0, 0, 0, 0
            for layer in model_pattern:
                if layer == "S":
                    S += 1  # noqa: N806
                elif layer == "D":
                    D += 1  # noqa: N806
                elif layer == "H":
                    H += 1  # noqa: N806
                elif layer == "*":
                    A += 1  # noqa: N806
            return S, D, H, A

        # Count S, D, H, and * layers in HyenaModel.
        S, D, H, A = _hyena_layer_count(self.hybrid_override_pattern)  # noqa: N806
        # Logits FLOPs per batch for a flattened L x H -> V GEMM.
        logits_fpl = 2 * batch_size * self.seq_length * self.hidden_size * self.vocab_size
        # Hyena Mixer Common FLOPs - Pre-Attention QKV Projections, Post-Attention Projections, and
        #   GLU FFN FLOPs per layer.
        pre_attn_qkv_proj_fpl = 2 * 3 * batch_size * self.seq_length * self.hidden_size**2
        post_attn_proj_fpl = 2 * batch_size * self.seq_length * self.hidden_size**2
        # 3 Batched GEMMs: y = A(gelu(Bx) * Cx) where B,C: H -> F and A: F -> H.
        glu_ffn_fpl = 2 * 3 * batch_size * self.seq_length * self.ffn_hidden_size * self.hidden_size
        # Transformer (Self) Attention FLOPs - QK Attention Logits ((L, D) x (D, L)) & Attention-Weighted
        #   Values FLOPs ((L, L) x (L, D))
        attn_fpl = 2 * 2 * batch_size * self.hidden_size * self.seq_length**2
        # Hyena Projection
        hyena_proj_fpl = 2 * 3 * batch_size * self.seq_length * hyena_short_conv_L * self.hidden_size
        # Hyena Short Conv
        hyena_short_conv_fpl = 2 * batch_size * self.seq_length * hyena_short_conv_len * self.hidden_size
        # Hyena Medium Conv
        hyena_medium_conv_fpl = 2 * batch_size * self.seq_length * hyena_medium_conv_len * self.hidden_size
        # Hyena Long Conv (FFT)
        hyena_long_conv_fft_fpl = batch_size * 10 * self.seq_length * math.log2(self.seq_length) * self.hidden_size
        # Based off of https://gitlab-master.nvidia.com/clara-discovery/savanna/-/blob/main/savanna/mfu.py#L182
        # Assumption: 1x Backwards Pass FLOPS = 2x Forward Pass FLOPS
        return 3 * (
            logits_fpl
            + self.num_layers * (pre_attn_qkv_proj_fpl + post_attn_proj_fpl + glu_ffn_fpl)
            + A * attn_fpl
            + (S + D + H) * hyena_proj_fpl
            + S * hyena_short_conv_fpl
            + D * hyena_medium_conv_fpl
            + H * hyena_long_conv_fft_fpl
        )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreHyenaModel:
        """Configures and returns a Hyena model instance based on the config settings.

        Args:
            pre_process: Whether to preprocess the inputs prior to running the rest of forward. Set to False if this is not the first stage of the pipeline.
            post_process: Whether to postprocess the outputs after running the rest of forward. Set to False if this is not the last stage of the pipeline, or if you are collecting hidden states.
            vp_stage: Virtual pipeline stage if using VPP and pipeline parallelism.

        Returns:
            MCoreHyenaModel: Configured Hyena model instance
        """
        self.bias_activation_fusion = False if self.remove_activation_post_first_layer else self.bias_activation_fusion

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in Hyena."
        )

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        model = MCoreHyenaModel(
            self,
            hyena_stack_spec=get_hyena_stack_spec(
                use_te=self.use_te,
                vortex_style_fp8=self.vortex_style_fp8,
                unfused_rmsnorm=self.unfused_rmsnorm,
                plain_row_linear=self.plain_row_linear,
            ),
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            num_groups_hyena=self.num_groups_hyena,
            num_groups_hyena_medium=self.num_groups_hyena_medium,
            num_groups_hyena_short=self.num_groups_hyena_short,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            # Note: When self.pre_process/self.post_process is explicitly False (e.g., for embedding
            # extraction), we must use that value regardless of what the caller passes. This is because
            # _create_model in megatron.bridge always passes the pipeline stage values, but we want to
            # disable post-processing when extracting embeddings.
            pre_process=(
                False
                if self.pre_process is False
                else (pre_process if pre_process is not None else parallel_state.is_pipeline_first_stage())
            ),
            post_process=(
                False
                if self.post_process is False
                else (post_process if post_process is not None else parallel_state.is_pipeline_last_stage())
            ),
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            hyena_init_method=self.hyena_init_method,
            hyena_output_layer_init_method=self.hyena_output_layer_init_method,
            remove_activation_post_first_layer=self.remove_activation_post_first_layer,
            add_attn_proj_bias=self.add_attn_proj_bias,
        )
        return model


@dataclass
class HyenaTestModelProvider(HyenaModelProvider):
    """Configuration for testing Hyena models."""

    hybrid_override_pattern: str = "SDH*"
    num_layers: int = 4
    seq_length: int = 8192
    hidden_size: int = 4096
    num_groups_hyena: int = 4096
    num_groups_hyena_medium: int = 256
    num_groups_hyena_short: int = 256
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = "byte-level"
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 2
    hyena_init_method: str = "small_init"
    hyena_output_layer_init_method: str = "wang_init"
    hyena_filter_no_wd: bool = True
    use_short_conv_bias: bool = False
    use_subquadratic_ops: bool = False


@dataclass
class HyenaNVTestModelProvider(HyenaTestModelProvider):
    """This config addresses several design improvements over the original implementation, and may provide better training stability for new models."""

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True


@dataclass
class Hyena1bModelProvider(HyenaModelProvider):
    """Config matching the 1b 8k context Evo2 model."""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*"
    num_layers: int = 25
    recompute_num_layers: int = 5  # needs to be a multiple of num_layers
    seq_length: int = 8192
    hidden_size: int = 1920
    num_groups_hyena: int = 1920
    num_groups_hyena_medium: int = 128
    num_groups_hyena_short: int = 128
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = "byte-level"
    mapping_type: str = "base"
    ffn_hidden_size: int = 5120
    gated_linear_unit: bool = True
    num_attention_heads: int = 15
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 5
    hyena_init_method: str = "small_init"
    hyena_output_layer_init_method: str = "wang_init"
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV1bModelProvider(Hyena1bModelProvider):
    """This config addresses several design improvements over the original implementation, and may provide better training stability for new models."""

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True


@dataclass
class Hyena7bModelProvider(HyenaModelProvider):
    """Config matching the 7b 8k context Evo2 model."""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*"
    num_layers: int = 32
    seq_length: int = 8192
    hidden_size: int = 4096
    num_groups_hyena: int = 4096
    num_groups_hyena_medium: int = 256
    num_groups_hyena_short: int = 256
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = "byte-level"
    mapping_type: str = "base"
    ffn_hidden_size: int = 11008
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 4
    hyena_init_method: str = "small_init"
    hyena_output_layer_init_method: str = "wang_init"
    hyena_filter_no_wd: bool = True


@dataclass
class HyenaNV7bModelProvider(Hyena7bModelProvider):
    """This config addresses several design improvements over the original implementation, and may provide better training stability for new models."""

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True
    ffn_hidden_size: int = 11264  # start with the larger FFN hidden size to avoid having to pad during extension.
    rotary_base: int = 1_000_000


@dataclass
class Hyena40bModelProvider(HyenaModelProvider):
    """Config matching the 40b 8k context Evo2 model."""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*SDH*SDHSDH*SDHSDH*"
    num_layers: int = 50
    seq_length: int = 8192
    hidden_size: int = 8192
    num_groups_hyena: int = 8192
    num_groups_hyena_medium: int = 512
    num_groups_hyena_short: int = 512
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = "byte-level"
    mapping_type: str = "base"
    ffn_hidden_size: int = 21888
    gated_linear_unit: bool = True
    num_attention_heads: int = 64
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 2
    hyena_init_method: str = "small_init"
    hyena_output_layer_init_method: str = "wang_init"
    hyena_filter_no_wd: bool = True
    rotary_base: int = 1_000_000


@dataclass
class HyenaNV40bModelProvider(Hyena40bModelProvider):
    """This config addresses several design improvements over the original implementation, and may provide better training stability for new models."""

    remove_activation_post_first_layer: bool = False
    add_attn_proj_bias: bool = False
    use_short_conv_bias: bool = True
    ffn_hidden_size: int = 22528  # start with the larger FFN hidden size to avoid having to pad during extension.


@dataclass
class Hyena7bARCLongContextModelProvider(Hyena7bModelProvider):
    """The checkpoint from ARC requires padding to the FFN dim due to requirements of large TP size for training at long context.

    NOTE: This config _could be used_ for short context as well with a different seq_length.
    """

    seq_length: int = 1_048_576
    ###
    # Hand verification of RoPE base for 7B-1M @antonvnv
    # >>> def inv_freq(base, dim):
    #     return 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
    #
    # >>> pt = torch.load("evo2_7b.pt", weights_only=False, mmap=True, map_location="cpu")
    #
    # >>> torch.mean(pt["blocks.3.inner_mha_cls.rotary_emb.inv_freq"] - inv_freq(1e11, 128).cpu())
    # tensor(0.0688)
    #
    # >>> torch.mean(pt["blocks.3.inner_mha_cls.rotary_emb.inv_freq"] - inv_freq(1e6, 128).cpu())
    # tensor(0.0361)
    #
    # >>> torch.mean(pt["blocks.3.inner_mha_cls.rotary_emb.inv_freq"] - inv_freq(1e4, 128).cpu())
    # tensor(5.2014e-06)
    rotary_base: int = 10_000
    ffn_hidden_size: int = 11264
    seq_len_interpolation_factor: float = 128


@dataclass
class Hyena40bARCLongContextModelProvider(Hyena40bModelProvider):
    """The checkpoint from ARC requires padding to the FFN dim due to requirements of large TP size for training at long context.

    NOTE: This config _could be used_ for short context as well with a different seq_length.
    """

    seq_length: int = 1_048_576
    ####
    # For 40B-1M hand verification of RoPE base @antonvnv
    # >>> def inv_freq(base, dim):
    #     return 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
    #
    # >>> pt = torch.load("evo2_40b.pt", weights_only=False, mmap=True, map_location="cpu")
    #
    # >>> torch.mean(pt["blocks.3.inner_mha_cls.rotary_emb.inv_freq"] - inv_freq(1e11, 128).cpu())
    # tensor(0.0326)
    #
    # >>> torch.mean(pt["blocks.3.inner_mha_cls.rotary_emb.inv_freq"] - inv_freq(1e6, 128).cpu())
    # tensor(-2.5294e-05)
    rotary_base: int = 1_000_000
    ffn_hidden_size: int = 22528
    seq_len_interpolation_factor: float = 128


@dataclass
class Hyena20bARCModelProvider(Hyena40bARCLongContextModelProvider):
    """Config matching the 20b 1M context Evo2 model (arcinstitute/evo2_20b).

    Architecturally identical to the 40b long-context model but with 24 layers
    instead of 50.  All other hyperparameters (hidden_size, num_attention_heads,
    ffn_hidden_size, rotary settings, etc.) are inherited from the 40b config.

    Source: evo2/configs/evo2-20b-1m.yml from ARC's evo2 repo.
    Layer pattern derived from: hcs=[0,4,7,11,14,18,21], hcm=[1,5,8,12,15,19,22],
    hcl=[2,6,9,13,16,20,23], attn=[3,10,17].
    """

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH"
    num_layers: int = 24


@dataclass
class HyenaNV1b2ModelProvider(HyenaNV1bModelProvider):
    """A parallel friendly version of the HyenaNV1bConfig."""

    hidden_size: int = 2048  # 1920
    num_groups_hyena: int = 2048  # 1920
    num_attention_heads: int = 16  # 15
    ffn_hidden_size: int = 5120  # 5120
    # Spike-no-more-embedding init by default.
    share_embeddings_and_output_weights: bool = False
    embedding_init_method_std: float = 1.0
    # activation_func_clamp_value: Optional[float] = 7.0
    # glu_linear_offset: float = 1.0


HYENA_MODEL_OPTIONS: dict[str, Type[HyenaModelProvider]] = {
    # ARC public checkpoint names (evo2_ prefix matches HuggingFace repo names)
    "evo2_1b_base": Hyena1bModelProvider,
    "evo2_7b_base": Hyena7bModelProvider,
    "evo2_7b": Hyena7bARCLongContextModelProvider,
    "evo2_20b": Hyena20bARCModelProvider,
    "evo2_40b_base": Hyena40bModelProvider,
    "evo2_40b": Hyena40bARCLongContextModelProvider,
    # NVIDIA-modified variants (striped_hyena_ prefix, no public ARC checkpoint)
    "striped_hyena_1b_nv": HyenaNV1bModelProvider,
    "striped_hyena_7b_nv": HyenaNV7bModelProvider,
    "striped_hyena_40b_nv": HyenaNV40bModelProvider,
    "striped_hyena_test": HyenaTestModelProvider,
    "striped_hyena_test_nv": HyenaNVTestModelProvider,
    "striped_hyena_1b_nv_parallel": HyenaNV1b2ModelProvider,
}


MODEL_OPTIONS = HYENA_MODEL_OPTIONS


def infer_model_type(model_size: str) -> str:
    """Infer the model architecture type from the model size key.

    Args:
        model_size: A model size key such as ``"evo2_1b_base"``.

    Returns:
        ``"hyena"`` if *model_size* is in :data:`HYENA_MODEL_OPTIONS`.

    Raises:
        ValueError: If the key is not found in any model options dict.
    """
    if model_size in HYENA_MODEL_OPTIONS:
        return "hyena"
    raise ValueError(f"Unknown model size: {model_size!r}. Valid options: {sorted(MODEL_OPTIONS.keys())}")


__all__ = [
    "HYENA_MODEL_OPTIONS",
    "MODEL_OPTIONS",
    "Hyena1bModelProvider",
    "Hyena7bARCLongContextModelProvider",
    "Hyena7bModelProvider",
    "Hyena20bARCModelProvider",
    "Hyena40bARCLongContextModelProvider",
    "Hyena40bModelProvider",
    "HyenaModelProvider",
    "HyenaNV1b2ModelProvider",
    "HyenaNV1bModelProvider",
    "HyenaNV7bModelProvider",
    "HyenaNV40bModelProvider",
    "HyenaNVTestModelProvider",
    "HyenaTestModelProvider",
    "infer_model_type",
]
