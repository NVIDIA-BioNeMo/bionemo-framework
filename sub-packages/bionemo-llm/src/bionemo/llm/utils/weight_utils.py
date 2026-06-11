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

from pathlib import Path
from typing import Sequence, Set

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from bionemo.llm.api import MegatronModelType


__all__: Sequence[str] = (
    "load_weights_sharded_inplace_nemo2_to_mcore",
    "nemo1_to_nemo2_biobert_key_mapping",
)


def nemo1_to_nemo2_biobert_key_mapping(  # noqa: D417
    old_key: str,
    new_model_prefix: str = "module",
    old_model_prefix: str = "model",
    te_mapping: bool = False,
) -> str:
    """This function is used to map the keys from the old nemo BERT models to the new BioBERT models

    Args:
        old_key (str): old key we want to map to the expected new key name.
        new_model_prefix (str, optional): The new key for the base weights.
            If you point this at the core megatron model set it to "".
            For the regular nemo2 lightning module following standards, set it to "module".
            Defaults to "module".
        old_model_prefix (str, optional): The previous saved weight prefix. Defaults to "model" which was the standard in nemo1.

    Returns:
        str: New key name
    """  # noqa: D415
    # add the . to the end of the input prefixes if they are not the empty string,
    #  unless the user has already done so.
    if old_model_prefix != "":
        old_model_prefix = f"{old_model_prefix.rstrip('.')}."
    if new_model_prefix != "":
        new_model_prefix = f"{new_model_prefix.rstrip('.')}."

    # This function is used to map the keys from the old nemo BERT models to the new BioBERT models
    base_rename = old_key.replace(f"{old_model_prefix}language_model.", f"{new_model_prefix}")
    base_rename = base_rename.replace(f"{old_model_prefix}", f"{new_model_prefix}")
    if "dense_h_to_4h" in base_rename:
        return base_rename.replace("dense_h_to_4h", "linear_fc1")
    if "dense_4h_to_h" in base_rename:
        return base_rename.replace("dense_4h_to_h", "linear_fc2")
    if "query_key_value" in base_rename:
        return base_rename.replace("query_key_value", "linear_qkv")
    if "self_attention.dense" in base_rename:
        #  This is definitely the linear_proj and not the qkv. The linear_proj shapes are 256x256
        #   which match dense but not query_key_value
        # (Pdb) new_state_dict['encoder.layers.4.self_attention.linear_proj.weight'].shape
        #  torch.Size([256, 256])
        # (Pdb) new_state_dict['encoder.layers.4.self_attention.linear_qkv.weight'].shape
        # torch.Size([768, 256])
        # (Pdb) new_state_dict['encoder.layers.4.self_attention.linear_qkv.bias'].shape
        # torch.Size([768])
        return base_rename.replace("self_attention.dense", "self_attention.linear_proj")
    if "lm_head.bias" in base_rename:
        return base_rename.replace("lm_head.bias", "output_layer.bias")
    if "lm_head.weight" in base_rename:
        return base_rename.replace("lm_head.weight", "output_layer.weight")
    if "lm_head.layernorm" in base_rename:
        return base_rename.replace("lm_head.layernorm", "lm_head.layer_norm")

    if "post_attention_layernorm" in base_rename:
        base_rename = base_rename.replace("post_attention_layernorm", "pre_mlp_layernorm")

    # Handle the transformer engine spec's differences in layer naming and where things like layernorm are stored.
    #  TE moves layernorm from  an object that's part of the main attention layer to being an internal component of
    #  the linear layers, probably for efficiency/fusion of some sort.
    if te_mapping:
        if ".input_layernorm.weight" in base_rename:
            return base_rename.replace(".input_layernorm.weight", ".self_attention.linear_qkv.layer_norm_weight")
        if ".input_layernorm.bias" in base_rename:
            return base_rename.replace(".input_layernorm.bias", ".self_attention.linear_qkv.layer_norm_bias")
        if ".pre_mlp_layernorm.bias" in base_rename:
            return base_rename.replace(".pre_mlp_layernorm.bias", ".mlp.linear_fc1.layer_norm_bias")
        if ".pre_mlp_layernorm.weight" in base_rename:
            return base_rename.replace(".pre_mlp_layernorm.weight", ".mlp.linear_fc1.layer_norm_weight")
    return base_rename


#############################################################################################
# Core utility functions: Below are some utility functions that allow for loading a nemo2
#  trained model back into a newly initialized megatron core model. The key insight is that
#  the nemo2 lightning module owns a single `self.module = config.configure_model(...)`
#  object. This `config.configure_module(...)` object is the megatron model that we want
#  to load weights into. So we need to adjust the checkpoint keys since they will all
#  have the extra `module.` prefix on them, while the megatron model we just initialized
#  will not. These functions should make a wide variety of fine-tuning strategies doable.


def _munge_key_megatron_to_nemo2(k: str) -> str:
    return f"module.{k}"


def _munge_sharded_tensor_key_megatron_to_nemo2(v: ShardedTensor) -> ShardedTensor:
    # This works with PP=1, how do we handle PP>1?
    key = v.key
    v.key = _munge_key_megatron_to_nemo2(key)
    return v


def _key_in_filter(k: str, filter: Set[str]) -> bool:
    for prefix in filter:
        if k.startswith(prefix):
            return True
    return False


def load_weights_sharded_inplace_nemo2_to_mcore(
    model: MegatronModelType, distributed_checkpoint_dir: str | Path, skip_keys_with_these_prefixes: Set[str]
) -> None:
    """Given a megatron module, this function will determine which keys/subsets of weights to load given the
        parallel/distributed state. This operates assuming a checkpoint was saved by a nemo2 trainer which places
        the `module.` prefix on all key names, but we are then going to load directly in to the megatron module
        without the `module.` prefix. Note that if there are any _extra_ keys that you do not want to search the
        checkpoint for, for example if you add new layers/heads onto your module, you need to supply the prefix
        path to those keys in your model and they will be ignored. This latter feature is key for flexible fine-tuning
        strategies where you load weights partially from other models with partially overlapping structures.

    Args:
        model: Megatron model that you want to load weights into.
        distributed_checkpoint_dir: _description_
        skip_keys_with_these_prefixes: _description_
    """  # noqa: D205
    sharded_state_dict = {
        _munge_key_megatron_to_nemo2(k): _munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not _key_in_filter(k, skip_keys_with_these_prefixes) and "_extra_state" not in k
    }
    dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=str(Path(distributed_checkpoint_dir) / "weights"),
        strict=dist_checkpointing.serialization.StrictHandling.ASSUME_OK_UNEXPECTED,
    )
