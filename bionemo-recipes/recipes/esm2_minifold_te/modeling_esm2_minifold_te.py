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

"""ESM2-MiniFold TE: End-to-end protein structure prediction model.

Combines a frozen HuggingFace ESM-2 backbone with a TE-based MiniFold folding head.
The ESM-2 backbone extracts per-residue embeddings and pairwise attention maps,
which are projected and fed into the FoldingTrunkTE for distogram prediction.

Optionally includes the StructureModuleTE for full 3D structure prediction (Stage 2).
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from esm_backbone import ESM2Backbone
from heads_te import AuxiliaryHeadsTE
from minifold_utils.feats import atom14_to_atom37
from minifold_utils.tensor_utils import tensor_tree_map
from model_te import FoldingTrunkTE, PairToSequenceTE
from quantization import ComponentPrecisionConfig
from structure_te import StructureModuleTE
from te_utils import te_linear_nd


class ESM2MiniFoldTE(nn.Module):
    """ESM-2 backbone + MiniFold TE folding head.

    Stage 1: ESM-2 (frozen) -> projections -> FoldingTrunkTE -> distogram predictions
    Stage 2: + PairToSequenceTE -> StructureModuleTE -> 3D coordinates + pLDDT
    """

    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        c_s: int = 1024,
        c_z: int = 128,
        num_blocks: int = 48,
        no_bins: int = 64,
        use_structure_module: bool = False,
        num_structure_blocks: int = 8,
        structure_config: dict | None = None,
        params_dtype: torch.dtype = torch.float32,
        block_precision: list[str | None] | None = None,
        fp8_recipe=None,
        fp4_recipe=None,
        component_precision: ComponentPrecisionConfig | None = None,
        te_module_device: str | torch.device | None = None,
    ):
        """Initialize ESM2MiniFoldTE.

        Args:
            esm_model_name: HuggingFace ESM-2 model name.
            c_s: Sequence feature dimension after projection.
            c_z: Pair feature dimension after projection.
            num_blocks: Number of MiniFormer blocks in the folding trunk.
            no_bins: Number of distogram bins.
            use_structure_module: Whether to include the structure module (Stage 2).
            num_structure_blocks: Number of IPA blocks in the structure module.
            structure_config: Optional config dict for auxiliary heads.
            params_dtype: Data type for TE layer parameters.
            block_precision: Per-block quantization precision list from resolve_layer_precision().
            fp8_recipe: FP8 recipe for TE autocast.
            fp4_recipe: FP4 recipe for TE autocast.
            component_precision: Per-component precision overrides within FP8/FP4 blocks.
            te_module_device: Optional device context used only while constructing
                the TE folding head. Set to ``"meta"`` to create TE params on the
                meta device before FSDP2 sharding/materialization.
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.use_structure_module = use_structure_module
        self._component_precision = component_precision

        # ESM-2 backbone (frozen)
        self.backbone = ESM2Backbone(esm_model_name)
        embed_dim = self.backbone.embed_dim
        attn_dim = self.backbone.attn_dim

        te_device_ctx = torch.device(te_module_device) if te_module_device is not None else nullcontext()
        with te_device_ctx:
            # Sequence projection: embed_dim -> c_s
            self.fc_s_1 = te.Linear(embed_dim, c_s, params_dtype=params_dtype)
            self.fc_s_2 = te.Linear(c_s, c_s, params_dtype=params_dtype)

            # Pairwise projection: attn_dim -> c_z
            self.fc_z_1 = te.Linear(attn_dim, c_z, params_dtype=params_dtype)
            self.fc_z_2 = te.Linear(c_z, c_z, params_dtype=params_dtype)

            # Folding trunk
            self.fold = FoldingTrunkTE(
                c_s=c_s,
                c_z=c_z,
                bins=32,
                disto_bins=no_bins,
                num_layers=num_blocks,
                params_dtype=params_dtype,
                block_precision=block_precision,
                fp8_recipe=fp8_recipe,
                fp4_recipe=fp4_recipe,
                component_precision=component_precision,
            )

            # Optional structure module (Stage 2)
            if use_structure_module:
                self.sz_project = PairToSequenceTE(c_z=c_z, c_s=c_s, params_dtype=params_dtype)
                self.structure_module = StructureModuleTE(
                    c_s=c_s,
                    c_z=c_z,
                    c_resnet=128,
                    head_dim=64,
                    no_heads=16,
                    no_blocks=num_structure_blocks,
                    no_resnet_blocks=2,
                    no_angles=7,
                    trans_scale_factor=10,
                    epsilon=1e-5,
                    inf=1e5,
                    params_dtype=params_dtype,
                )
                if structure_config is not None:
                    self.aux_heads = AuxiliaryHeadsTE(structure_config["heads"])

    def forward(self, batch: dict, num_recycling: int = 0) -> dict:
        """Forward pass.

        Args:
            batch: Dictionary with:
                "input_ids": ESM-2 token IDs (B, L).
                "attention_mask": Optional attention mask (B, L).
                "mask": Residue validity mask (B, L).
                "batch_of": Optional OpenFold features for structure module.
            num_recycling: Number of recycling iterations.

        Returns:
            Dictionary with predictions:
                "preds": Distogram logits (B, L, L, no_bins).
                "pair": Final pair representation (B, L, L, c_z).
                + structure module outputs if use_structure_module.
        """
        # Extract ESM-2 embeddings and attention maps
        esm_out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        # Project sequence embeddings: embed_dim -> c_s
        cp = self._component_precision
        seq_ctx = cp.get_context("seq_proj") if cp else nullcontext()
        with seq_ctx:
            s_s = esm_out["representations"]
            s_s = te_linear_nd(self.fc_s_1, s_s)
            s_s = torch.relu(s_s)
            s_s = te_linear_nd(self.fc_s_2, s_s)

            # Project attention maps: attn_dim -> c_z
            s_z = esm_out["attentions"]
            s_z = te_linear_nd(self.fc_z_1, s_z)
            s_z = torch.relu(s_z)
            s_z = te_linear_nd(self.fc_z_2, s_z)

        # Run folding trunk
        preds, s_z = self.fold(
            s_s,
            s_z,
            mask=batch["mask"],
            num_recycling=num_recycling,
        )

        r_dict = {"preds": preds, "pair": s_z}

        # Optional structure module
        if self.use_structure_module:
            mask = batch["mask"]
            pair_mask = mask[:, None, :] * mask[:, :, None]
            r_dict["single"] = self.sz_project(s_z, s_s, pair_mask)

            feats = tensor_tree_map(lambda t: t[..., 0], batch["batch_of"])

            r_dict["sm"] = self.structure_module(
                s=r_dict["single"],
                z=r_dict["pair"],
                aatype=feats["aatype"],
                mask=feats["seq_mask"].to(dtype=r_dict["single"].dtype),
            )

            r_dict["final_atom_positions"] = atom14_to_atom37(r_dict["sm"]["positions"][-1], feats)
            r_dict["final_atom_mask"] = feats["atom37_atom_exists"]
            r_dict["final_affine_tensor"] = r_dict["sm"]["frames"][-1]

            if hasattr(self, "aux_heads"):
                r_dict.update(self.aux_heads(r_dict))

        return r_dict

    def get_folding_head_params(self):
        """Return parameters for the folding head (for optimizer param groups)."""
        excluded = {"backbone", "structure_module", "aux_heads", "sz_project"}
        for name, param in self.named_parameters():
            if not any(name.startswith(prefix) for prefix in excluded):
                if param.requires_grad:
                    yield param

    def get_structure_module_params(self):
        """Return parameters for the structure module (for optimizer param groups)."""
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in ("structure_module", "aux_heads", "sz_project")):
                if param.requires_grad:
                    yield param

    def get_backbone_params(self):
        """Return backbone parameters (typically frozen)."""
        return self.backbone.parameters()
