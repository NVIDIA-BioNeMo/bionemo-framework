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


import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from src.data.metadata import MetadataFields
from src.inference.base import BaseInference
from src.inference.model_outputs import (
    DownstreamPredictionOutput,
    EmbeddingOutput,
    FitnessPredictionOutput,
    MaskedLMOutput,
    MutationPredictionOutput,
)
from src.inference.task_types import TaskTypes
from src.models.encodon_pl import EncodonPL
from src.models.encodon_te_pl import EncodonTEPL
from src.tokenizer import Tokenizer
from src.utils.load_checkpoint import resolve_model_path


class EncodonInference(BaseInference):
    """Inference class for Encodon models."""

    def configure_model(self):
        """Loads the model and tokenizer for inference."""
        if self.model is not None:
            return
        self.tokenizer = Tokenizer()

        state_dict = None
        hparams = None

        # Resolve model_path: local file/dir pass through, Hub repo IDs are downloaded.
        model_path = Path(resolve_model_path(self.model_path))

        if model_path.is_dir():
            # Load from safetensors format
            safetensors_files = list(model_path.glob("*.safetensors"))

            if not safetensors_files or len(safetensors_files) != 1:
                raise FileNotFoundError(f"Expected single .safetensors files in {model_path}")

            safetensors_path = safetensors_files[0]
            config_path = model_path / "config.json"

            print(f"Loading safetensors from: {safetensors_path}")
            state_dict = load_file(str(safetensors_path))
            with open(config_path, "r") as f:
                hparams = json.load(f)
        else:
            # Load from PyTorch Lightning checkpoint
            if dist.is_initialized():
                broadcasted_objects = [None, None]

                if dist.get_rank() == 0:
                    ckpt = torch.load(self.model_path, map_location="cpu")
                    hparams = ckpt.get("hyper_parameters")
                    state_dict = ckpt.get("state_dict")
                    broadcasted_objects = [state_dict, hparams]

                dist.broadcast_object_list(broadcasted_objects, src=0)
                state_dict, hparams = broadcasted_objects
            else:
                ckpt = torch.load(self.model_path, map_location="cpu")
                hparams = ckpt.get("hyper_parameters")
                state_dict = ckpt.get("state_dict")

        # The hparams from lightning checkpoint might be nested.
        if "hparams" in hparams:
            hparams = hparams["hparams"]

        # For inference we don't need optimizer and scheduler, but EncodonPL expects them.
        def dummy_optimizer(params):
            return torch.optim.Adam(params)

        hparams["optimizer"] = dummy_optimizer
        hparams["scheduler"] = None

        hparams["attn_input_format"] = self.attn_input_format

        self.model = EncodonTEPL(**hparams) if self.use_transformer_engine else EncodonPL(**hparams)
        self.model.configure_model(state_dict=state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_mlm(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """Predicts masked tokens in a batch."""
        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            if preds.dtype != torch.float:
                preds = preds.float()
            mask = batch[MetadataFields.INPUT_MASK].bool()
            y = batch[MetadataFields.LABELS]
            y = y[mask]
            preds = preds[mask]
            preds = preds.cpu().numpy()
            if len(y) > 0:
                y = y.cpu().numpy()

        return MaskedLMOutput(preds=preds, labels=y, ids=ids)

    def predict_mutation(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """Predicts the effect of a mutation by calculating the log-likelihood ratio
        of the reference codon vs. the alternative codon at the mutation site.
        """  # noqa: D205
        # breakpoint()
        # input_ids = torch.from_numpy(np.load('input_ids.npy')).cuda()
        # ref_codon_toks = torch.from_numpy(np.load('ref_codon_toks.npy')).cuda()
        # alt_codon_toks = torch.from_numpy(np.load('alt_codon_toks.npy')).cuda()
        # mutation_token_idx = torch.from_numpy(np.load('mutation_token_idx.npy')).cuda()
        # attention_mask = torch.from_numpy(np.load('attention_mask.npy')).cuda()
        # ids = torch.from_numpy(np.load('ids.npy')).cuda()

        # batch = {
        #     "input_ids": input_ids,
        #     "ref_codon_toks": ref_codon_toks,
        #     "alt_codon_toks": alt_codon_toks,
        #     "mutation_token_idx": mutation_token_idx,
        #     "attention_mask": attention_mask,
        # }

        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            if preds.dtype != torch.float:
                preds = preds.float()
            ref_toks = batch[MetadataFields.REF_CODON_TOKS]
            alt_toks = batch[MetadataFields.ALT_CODON_TOKS]
            mutation_token_idx = batch[MetadataFields.MUTATION_TOKEN_IDX].view(-1)
            # Get predictions only for the mutated token positions.
            preds = preds[torch.arange(preds.shape[0]), mutation_token_idx, :]
            # Convert logits to log-probabilities.
            preds = torch.nn.functional.log_softmax(preds, dim=-1)
            # Get the log-likelihoods for the reference and alternate codons.
            ref_likelihoods = preds[torch.arange(preds.shape[0]), ref_toks]
            alt_likelihoods = preds[torch.arange(preds.shape[0]), alt_toks]
            # The likelihood ratio is the difference in log-likelihoods.
            likelihood_ratios = ref_likelihoods - alt_likelihoods
        return MutationPredictionOutput(
            ref_likelihoods=ref_likelihoods.cpu().numpy(),
            alt_likelihoods=alt_likelihoods.cpu().numpy(),
            likelihood_ratios=likelihood_ratios.cpu().numpy(),
            ids=ids,
        )

    def extract_embeddings(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """Extracts embeddings for a batch of sequences."""
        with torch.no_grad():
            output = self.model(batch, return_hidden_states=True)
            embeddings = output.all_hidden_states[-1]
            if embeddings.dtype != torch.float:
                embeddings = embeddings.float()
            if self.attn_input_format == "thd":
                cu_seq_lens = batch[MetadataFields.CU_SEQ_LENS_Q]
                cls_indices = cu_seq_lens[:-1].long()
                embeddings = embeddings[cls_indices]
            else:
                embeddings = embeddings[:, 0, :]  # [CLS] token
            embeddings = embeddings.cpu().numpy()
        return EmbeddingOutput(embeddings=embeddings, ids=ids)

    def predict_fitness(self, batch, ids=None) -> Dict[str, np.ndarray]:
        """Predicts a fitness score for each sequence in the batch.
        The fitness score is defined as the average log-likelihood of the sequence.
        """  # noqa: D205
        with torch.no_grad():
            output = self.model(batch)
            preds = output.logits
            if preds.dtype != torch.float:
                preds = preds.float()

            # Get log-probabilities for all tokens in the vocabulary.
            log_probs = torch.nn.functional.log_softmax(preds, dim=-1)
            # Gather the log-probabilities of the input tokens.
            selected_log_probs = log_probs.gather(-1, batch[MetadataFields.INPUT_IDS].unsqueeze(-1)).squeeze(-1)
            # Create a mask to exclude padding tokens from the calculation.
            non_padding_mask = batch[MetadataFields.INPUT_IDS] != self.tokenizer.pad_token_id
            # Apply the mask to zero out log-probabilities of padding tokens.
            masked_log_probs = selected_log_probs * non_padding_mask
            # Sum the log-likelihoods for each sequence, ignoring padding.
            log_likelihoods_sum = masked_log_probs.sum(dim=-1)
            # Count the number of non-padding tokens in each sequence.
            non_padding_counts = non_padding_mask.sum(dim=-1)
            # Compute the mean log-likelihood per sequence.
            log_likelihoods_mean = (log_likelihoods_sum / non_padding_counts).cpu().numpy()
        return FitnessPredictionOutput(fitness=log_likelihoods_mean, ids=ids)

    def predict_downstream(self, batch, ids=None) -> DownstreamPredictionOutput:
        """Predicts using the downstream cross-attention head (classification or regression).
        This works with models that have use_downstream_head=True.
        """  # noqa: D205
        with torch.no_grad():
            # Check if model has downstream heads
            if not hasattr(self.model.model, "cross_attention_head") or not hasattr(
                self.model.model, "cross_attention_input_proj"
            ):
                raise ValueError(
                    "Model does not have downstream cross-attention heads. Ensure the model was trained with use_downstream_head=True."
                )

            # Get the base model output (hidden states)
            output = self.model(batch)
            hidden_states = output.last_hidden_state  # [batch_size, seq_len, hidden_size]
            attention_mask = batch[MetadataFields.ATTENTION_MASK]  # [batch_size, seq_len]

            # Project hidden states to cross-attention dimension
            projected_states = self.model.model.cross_attention_input_proj(
                hidden_states
            )  # [batch_size, seq_len, cross_attn_hidden_dim]

            # Extract [CLS] token as query and use full sequence as key/value
            query_input = projected_states[:, 0, :]  # [batch_size, cross_attn_hidden_dim]
            key_value_input = projected_states  # [batch_size, seq_len, cross_attn_hidden_dim]

            # Pass through cross-attention head
            preds = self.model.model.cross_attention_head(query_input, key_value_input, attention_mask)

            # Determine task type based on model configuration
            loss_type = getattr(self.model.hparams, "loss_type", "regression")

            if loss_type == "classification":
                # Classification: preds shape [batch_size, num_classes]
                preds = preds.float().cpu().numpy()

                # Get probabilities (softmax) and predicted classes
                probabilities = torch.nn.functional.softmax(torch.from_numpy(preds), dim=-1).numpy()
                predicted_classes = np.argmax(preds, axis=-1)

                return DownstreamPredictionOutput(
                    predictions=preds, probabilities=probabilities, predicted_classes=predicted_classes, ids=ids
                )
            else:
                # Regression: preds shape [batch_size, 1] -> squeeze to [batch_size]
                preds = preds.squeeze(-1).float().cpu().numpy()

                return DownstreamPredictionOutput(predictions=preds, ids=ids)

    def _predict_step(self, batch, batch_idx):
        """A single prediction step that dispatches to the correct prediction function
        based on the task type.
        """  # noqa: D205
        # - remove id column from batch if present
        #  (this is needed for the model forward to work correctly)
        ids = None
        if MetadataFields.ID in batch:
            ids = batch[MetadataFields.ID]
            del batch[MetadataFields.ID]
        # - set predict function based on task type
        if self.task_type == TaskTypes.MUTATION_PREDICTION:
            predict = self.predict_mutation
        elif self.task_type == TaskTypes.MASKED_LANGUAGE_MODELING:
            predict = self.predict_mlm
        elif self.task_type == TaskTypes.FITNESS_PREDICTION:
            predict = self.predict_fitness
        elif self.task_type == TaskTypes.EMBEDDING_PREDICTION:
            predict = self.extract_embeddings
        elif self.task_type == TaskTypes.DOWNSTREAM_PREDICTION:
            predict = self.predict_downstream
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        outputs = predict(batch, ids)
        return outputs
