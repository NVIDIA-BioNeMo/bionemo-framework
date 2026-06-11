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

# Copyright 2019 Ross Wightman (huggingface/pytorch-image-models)
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
#
#    MIT License
#
#    Copyright (c) 2020 Phil Wang (lucidrains/vit-pytorch)
#    Copyright (c) 2020 Andrej Karpathy (karpathy/minGPT)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
"""
Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * Class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman

Derived from code written by Ross Wightman (@rwightman)
and modified for demonstrative use by NVIDIA (@cspades).
"""

import math
import warnings
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    OrderedDict,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from transformer_engine.pytorch import TransformerLayer

    _TE_INSTALLED = True
except ImportError:
    _TE_INSTALLED = False


def build_vit_model(cfg, device_mesh=None, meta_init=False):
    """
    Build a ViT.

    Args:
        cfg: Hydra config.
        device_mesh: Device mesh. Only needed for TransformerEngine.

    Returns:
        model: The ViT model.
    """
    with (
        # Meta Device Initialization
        torch.device("meta") if meta_init else nullcontext()
    ):
        vit_kwargs = dict(cfg.model.vit)
        if meta_init:
            vit_kwargs["weight_init"] = None
        if cfg.model.transformer_engine and _TE_INSTALLED:
            assert device_mesh is not None, "[build_model] device_mesh is required when using TransformerEngine."
            vit_kwargs["block_fn"] = TransformerLayer
            vit_kwargs["micro_batch_size"] = cfg.dataset.train.batch_size
            vit_kwargs["tp_group"] = device_mesh["tp"].get_group()
            vit_kwargs["tp_size"] = device_mesh["tp"].size()
        model = VisionTransformer(**vit_kwargs)
        if cfg.model.channels_last:
            model.to(memory_format=torch.channels_last)
    # Return the model.
    return model


class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(self.init_values * torch.ones(dim))

    def reset_parameters(self):
        """Reset model parameters. Required method for Megatron-FSDP meta device initialization."""
        self.gamma.data.fill_(self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        fused_attn: bool = False,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if qk_norm or scale_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Optional[Type[nn.Module]] = None,
        fused_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert not scale_attn_norm and not scale_mlp_norm, "Scale norms not supported"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer("qkv_bias", None)
            self.register_parameter("mlp_bias", None)
        else:
            self.register_buffer("qkv_bias", torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim, bias=proj_bias)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class ParallelThingsBlock(nn.Module):
    """Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_parallel: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim)),
                            (
                                "attn",
                                Attention(
                                    dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_norm=qk_norm,
                                    scale_norm=scale_attn_norm,
                                    proj_bias=proj_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    norm_layer=norm_layer,
                                ),
                            ),
                            (
                                "ls",
                                LayerScale(dim, init_values=init_values) if init_values else nn.Identity(),
                            ),
                            (
                                "drop_path",
                                DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                            ),
                        ]
                    )
                )
            )
            self.ffns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim)),
                            (
                                "mlp",
                                mlp_layer(
                                    dim,
                                    hidden_features=int(dim * mlp_ratio),
                                    act_layer=act_layer,
                                    norm_layer=norm_layer if scale_mlp_norm else None,
                                    bias=proj_bias,
                                    drop=proj_drop,
                                ),
                            ),
                            (
                                "ls",
                                LayerScale(dim, init_values=init_values) if init_values else nn.Identity(),
                            ),
                            (
                                "drop_path",
                                DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                            ),
                        ]
                    )
                )
            )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None:
            attn_out = []
            for attn in self.attns:
                x_attn = attn.norm(x)
                x_attn = attn.attn(x_attn, attn_mask=attn_mask)
                x_attn = attn.ls(x_attn)
                x_attn = attn.drop_path(x_attn)
                attn_out.append(x_attn)
            x = x + torch.stack(attn_out).sum(dim=0)
        else:
            x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    output_fmt: Format

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = (img_size, img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


def patch_dropout_forward(
    x: torch.Tensor,
    prob: float,
    num_prefix_tokens: int,
    ordered: bool,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Common forward logic for patch dropout.

    Args:
        x: Input tensor of shape (B, L, D)
        prob: Dropout probability
        num_prefix_tokens: Number of prefix tokens to preserve
        ordered: Whether to maintain patch order
        training: Whether in training mode

    Returns:
        Tuple of (output tensor, keep_indices or None)
    """
    if not training or prob == 0.0:
        return x, None

    if num_prefix_tokens:
        prefix_tokens, x = x[:, :num_prefix_tokens], x[:, num_prefix_tokens:]
    else:
        prefix_tokens = None

    B = x.shape[0]
    L = x.shape[1]
    D = x.shape[2]
    # Randomly drop patches / tiles with probability prob.
    num_keep = max(1, int(L * (1.0 - prob)))
    keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]

    if ordered:
        # NOTE does not need to maintain patch order in typical transformer use,
        # but possibly useful for debug / visualization
        keep_indices = keep_indices.sort(dim=-1)[0]

    x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1, *x.shape[2:])))

    if x.shape[1] < L:
        # If the number of patches is not the same as the original sequence length,
        # we need to extend the sequence length to L again. This makes it easy to
        # use Transformer layers that expect a consistent sequence length while
        # still enabling patch dropout. Because patch order does not matter, i.e.
        # the ViT is a full-attention model, we concatenate to the end.
        x = torch.cat([x, torch.zeros(B, L - x.shape[1], D, device=x.device)], dim=1)

    if prefix_tokens is not None:
        x = torch.cat((prefix_tokens, x), dim=1)

    return x, keep_indices


class PatchDropout(nn.Module):
    """Patch Dropout without returning indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
        self,
        prob: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = patch_dropout_forward(x, self.prob, self.num_prefix_tokens, self.ordered, self.training)
        return output


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    lower_cdf = norm_cdf((a - mean) / std)
    upper_cdf = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * lower_cdf - 1, 2 * upper_cdf - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def trunc_normal_tf_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor


class AttentionPoolLatent(nn.Module):
    """Attention pooling w/ latent query"""

    fused_attn: torch.jit.Final[bool]

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        feat_size: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: Optional[int] = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[Type[nn.Module]] = None,
        act_layer: Optional[Type[nn.Module]] = nn.GELU,
        drop: float = 0.0,
        fused_attn: bool = False,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim**-0.5
        self.pool = pool_type
        self.fused_attn = fused_attn

        if pos_embed == "abs":
            assert feat_size is not None
            self.pos_embed = nn.Parameter(torch.zeros(feat_size, in_features))
        else:
            self.pos_embed = None

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, embed_dim))

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        if qk_norm:
            qk_norm_layer = norm_layer or nn.LayerNorm
            self.q_norm = qk_norm_layer(self.head_dim)
            self.k_norm = qk_norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm = norm_layer(out_features) if norm_layer is not None else nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio), act_layer=act_layer)

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
        trunc_normal_tf_(self.latent, std=self.latent_dim**-0.5)

    def reset_parameters(self):
        """Reset model parameters. Required method for Megatron-FSDP meta device initialization."""
        self.init_weights()

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x


def feature_take_indices(
    num_features: int,
    indices: Optional[Union[int, List[int]]] = None,
    as_set: bool = False,
) -> Tuple[List[int], int]:
    """Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forward() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        assert 0 < indices <= num_features, f"last-n ({indices}) is out of range (1 to {num_features})"
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            assert 0 <= idx < num_features, f"feature index {idx} is out of range (0 to {num_features - 1})"
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == "avg":
            x = x.mean(dim=1)
        elif pool_type == "avgmax":
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == "max":
            x = x.amax(dim=1)
        else:
            assert not pool_type, f"Unknown pool type {pool_type}"

    return x


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Recursively apply a function to all sub-modules in a module."""

    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        joined_child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=joined_child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    """Initialize a tensor with a variance scaling initialization."""

    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    """Initialize a tensor with a LeCun normal initialization."""

    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility).

    Args:
        module: Module to initialize.
        name: Module name for context.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = "", head_bias: float = 0.0) -> None:
    """ViT weight initialization, matching JAX (Flax) impl.

    Args:
        module: Module to initialize.
        name: Module name for context.
        head_bias: Bias value for head layer.
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if "mlp" in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed.

    Args:
        module: Module to initialize.
        name: Module name for context.
    """
    if isinstance(module, nn.Linear):
        if "qkv" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def get_init_weights_vit(mode: str = "jax", head_bias: float = 0.0) -> Callable:
    if "jax" in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif "moco" in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


class PosEmbed(nn.Module):
    """Module that applies the position embedding in the ViT."""

    def __init__(
        self,
        embed_dim: int,
        embed_len: int,
        pos_drop_rate: float,
        no_embed_class: bool = False,
        num_prefix_tokens: int = 1,
        cls_token: bool = True,
        reg_tokens: int = 0,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.no_embed_class = no_embed_class
        self.num_prefix_tokens = num_prefix_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)

    def reset_parameters(self):
        """Reset model parameters. Required method for Megatron-FSDP meta device initialization."""
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input."""
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if to_cat:
                x = torch.cat([*to_cat, x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat([*to_cat, x], dim=1)
            x = x + self.pos_embed

        return self.pos_drop(x)


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Optional[Literal["jax", "jax_nlhb", "moco", "timm"]] = "timm",
        init_variance_rescale: bool = False,
        embed_layer: Callable = PatchEmbed,
        embed_norm_layer: Optional[torch.nn.Module] = None,
        norm_layer: Optional[torch.nn.Module] = nn.LayerNorm,
        act_layer: Optional[torch.nn.Module] = nn.GELU,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        micro_batch_size: int = 1,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_size: int = 1,
    ) -> None:
        """Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of image input channels.
        num_classes: Number of classes for classification head.
        global_pool: Type of global pooling for final sequence (default: 'token').
        embed_dim: Transformer embedding dimension.
        depth: Depth of transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Enable bias for qkv projections if True.
        init_values: Layer-scale init values (layer-scale enabled if not None).
        class_token: Use class token.
        pos_embed: Use learnable position embeddings.
        no_embed_class: Don't include position embeddings for class (or reg) tokens.
        reg_tokens: Number of register tokens.
        pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
        final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
        fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
        drop_rate: Head dropout rate.
        pos_drop_rate: Position embedding dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        weight_init: Weight initialization scheme. Defaults to "timm".
        init_variance_rescale: Apply weight initialization fix (scaling w/ layer index) to control initial variance of input propagating through the model.
        embed_layer: Patch embedding layer.
        embed_norm_layer: Normalization layer to use / override in patch embed module.
        norm_layer: Normalization layer.
        act_layer: MLP activation layer.
        block_fn: Transformer block layer.
        micro_batch_size: Micro batch size for TE.
        tp_group: Tensor parallel group.
        tp_size: Tensor parallel size.
        """
        super().__init__()
        assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool in ("avg", "avgmax", "max") if fc_norm is None else fc_norm

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix

        embed_args = {}
        if embed_norm_layer is not None:
            embed_args["norm_layer"] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, "feat_ratio") else patch_size
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = (
            PosEmbed(
                embed_dim=embed_dim,
                embed_len=embed_len,
                pos_drop_rate=pos_drop_rate,
                no_embed_class=no_embed_class,
                num_prefix_tokens=self.num_prefix_tokens,
                cls_token=class_token,
                reg_tokens=reg_tokens,
            )
            if pos_embed
            else None
        )
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth, device="cpu")]  # stochastic depth decay rule

        self.block_fn = block_fn
        if _TE_INSTALLED and block_fn == TransformerLayer:
            self.blocks = nn.Sequential(
                *[
                    TransformerLayer(
                        hidden_size=embed_dim,
                        ffn_hidden_size=int(embed_dim * mlp_ratio),
                        num_attention_heads=num_heads,
                        hidden_dropout=drop_rate,
                        attention_dropout=attn_drop_rate,
                        layer_number=i + 1,
                        self_attn_mask_type="no_mask",
                        window_size=(-1, -1),
                        tp_group=tp_group,
                        tp_size=tp_size,
                        seq_length=embed_len,
                        micro_batch_size=micro_batch_size,
                        layer_type="encoder",
                        fuse_qkv_params=True,
                        activation="gelu",
                        attn_input_format="bshd",
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.Sequential(
                *[
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        scale_attn_norm=scale_attn_norm,
                        scale_mlp_norm=scale_mlp_norm,
                        proj_bias=proj_bias,
                        init_values=init_values,
                        proj_drop=proj_drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        mlp_layer=mlp_layer,
                    )
                    for i in range(depth)
                ]
            )
        self.feature_info = [
            {"module": f"blocks.{i}", "num_chs": embed_dim, "reduction": reduction} for i in range(depth)
        ]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init is not None:
            self.init_weights(weight_init)
        if init_variance_rescale:
            self.rescale_init_variance()

    def rescale_init_variance(self) -> None:
        """Apply weight initialization fix (scaling w/ layer index)."""

        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if _TE_INSTALLED and self.block_fn == TransformerLayer:
                rescale(layer.self_attention.proj.weight.data, layer_id + 1)
                rescale(layer.layernorm_mlp.fc2_weight.data, layer_id + 1)
            else:
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = "") -> None:
        """Initialize model weights.

        Args:
            mode: Weight initialization mode ('jax', 'jax_nlhb', 'moco', or '').
        """
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            self.pos_embed.init_weights()
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def reset_parameters(self):
        """Reset model parameters. Required method for Megatron-FSDP meta device initialization."""
        self.init_weights()

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "avgmax", "max", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map" and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        output_dict: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            output_dict: Return outputs as a dictionary with 'image_features' and 'image_intermediates' keys
            attn_mask: Optional attention mask for masked attention (e.g., for NaFlex)

        Returns:
            A tuple with (final_features, intermediates), a list of intermediate features, or a dictionary containing
            'image_features' and 'image_intermediates' (and optionally 'image_intermediates_prefix')
        """
        assert output_fmt in ("NCHW", "NLC"), "Output format must be one of NCHW or NLC."
        reshape = output_fmt == "NCHW"
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[: max_index + 1]
        for i, blk in enumerate(blocks):
            if attn_mask is not None:
                x = blk(x, attn_mask=attn_mask)
            else:
                x = blk(x)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0 : self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens :] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            intermediates = [y.reshape(B, height, width, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        # For dictionary output, handle prefix tokens separately
        if output_dict:
            result_dict = {}
            # Intermediates are always included
            result_dict["image_intermediates"] = intermediates
            if prefix_tokens is not None and return_prefix_tokens:
                result_dict["image_intermediates_prefix"] = prefix_tokens

            # Only include features if not intermediates_only
            if not intermediates_only:
                x_final = self.norm(x)
                result_dict["image_features"] = x_final

            return result_dict

        # For non-dictionary output, maintain the original behavior
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[: max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, "")
        return take_indices

    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        """Apply pooling to feature tokens.

        Args:
            x: Feature tensor.
            pool_type: Pooling type override.

        Returns:
            Pooled features.
        """
        if self.attn_pool is not None:
            if not self.pool_include_prefix:
                x = x[:, self.num_prefix_tokens :]
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier.

        Returns:
            Output tensor.
        """
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x
