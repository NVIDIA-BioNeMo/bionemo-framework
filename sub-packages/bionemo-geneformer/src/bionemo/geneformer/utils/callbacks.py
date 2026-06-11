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


import logging
import os
from typing import Any, Literal, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.lightning import batch_collator


IntervalT = Literal["epoch", "batch"]


class GeneformerPredictionWriter(BasePredictionWriter, pl.Callback):
    """A callback that writes predictions to disk at specified intervals during training."""

    def __init__(
        self,
        output_dir: str | os.PathLike,
        write_interval: IntervalT,
        tokenizer: GeneTokenizer,
        batch_dim_key_defaults: dict[str, int] | None = None,
        seq_dim_key_defaults: dict[str, int] | None = None,
        include_gene_embeddings: bool = False,
    ):
        """Initializes the callback.

        Args:
            output_dir: The directory where predictions will be written.
            write_interval: The interval at which predictions will be written. (batch, epoch)
            tokenizer: The GeneTokenizer instance for mapping input_ids to gene names, and filtering out special tokens.
            batch_dim_key_defaults: The default batch dimension for each key, if different from the standard 0.
            seq_dim_key_defaults: The default sequence dimension for each key, if different from the standard 1.
            include_gene_embeddings: Whether to include gene embeddings in the output predictions.
        """
        super().__init__(write_interval)
        self.output_dir = str(output_dir)
        self.include_gene_embeddings = include_gene_embeddings
        self.batch_dim_key_defaults = batch_dim_key_defaults
        self.seq_dim_key_defaults = seq_dim_key_defaults
        self.tokenizer = tokenizer

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Any,
        batch_indices: Sequence[int],
    ) -> None:
        """Writes predictions to disk at the end of each epoch.

        Writing all predictions on epoch end is memory intensive. It is recommended to use the batch writer instead for
        large predictions.

        Multi-device predictions will likely yield predictions in an order that is inconsistent with single device predictions and the input data.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance, required by PyTorch Lightning.
            predictions: The predictions made by the model.
            batch_indices: The indices of the batch, required by PyTorch Lightning.

        Raises:
            Multi-GPU predictions are output in an inconsistent order with multiple devices.
        """
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank

        result_path = os.path.join(self.output_dir, f"predictions__rank_{trainer.global_rank}.pt")

        # collate multiple batches / ignore empty ones
        collate_kwargs = {}
        if self.batch_dim_key_defaults is not None:
            collate_kwargs["batch_dim_key_defaults"] = self.batch_dim_key_defaults
        if self.seq_dim_key_defaults is not None:
            collate_kwargs["seq_dim_key_defaults"] = self.seq_dim_key_defaults

        prediction = batch_collator([item for item in predictions if item is not None], **collate_kwargs)

        # batch_indices is not captured due to a lightning bug when return_predictions = False
        # we use input IDs in the prediction to map the result to input

        if self.include_gene_embeddings and "input_ids" in prediction and "hidden_states" in prediction:
            hidden_states = prediction["hidden_states"]
            input_ids = prediction["input_ids"]

            logging.info("Calculating gene embeddings.")
            logging.info(f"hidden_states: {hidden_states.shape[:2]}; input_ids: {input_ids.shape[:2]}")
            assert hidden_states.shape[:2] == input_ids.shape[:2]

            # accumulators for calculating mean embedding for each input_id
            gene_embedding_accumulator = {}
            input_id_count = {}
            ensembl_IDs = {}
            gene_symbols = {}

            # iterate over all cells
            cell_count = len(input_ids)
            for i in range(cell_count):
                cell_state = hidden_states[i]
                cell_input_ids = input_ids[i].cpu().numpy()

                # iterate over each gene in the cell
                for idx, embedding in zip(cell_input_ids, cell_state):
                    # skip calculation for special tokens like [CLS], [SEP], [PAD], [MASK], [UKW]
                    if idx in self.tokenizer.all_special_ids:
                        continue

                    # accumulate embedding sum and count
                    if idx not in gene_embedding_accumulator:
                        # initialize embedding sum with first found embedding
                        gene_embedding_accumulator[idx] = embedding

                        # increment input_id count
                        input_id_count[idx] = 1
                    else:
                        # accumulate embedding sum
                        gene_embedding_accumulator[idx] += embedding

                        # increment input_id count
                        input_id_count[idx] += 1

            # divide each embedding sum by the total occurences of each gene to get an average
            for input_id in gene_embedding_accumulator.keys():
                gene_embedding_accumulator[input_id] /= input_id_count[input_id]
                # map input_ids to gene symbols and ensembl IDs
                ensembl_IDs[input_id] = self.tokenizer.decode_vocab.get(input_id, f"UNKNOWN_{input_id}")
                gene_symbols[input_id] = self.tokenizer.ens_to_gene.get(ensembl_IDs[input_id], f"UNKNOWN_{input_id}")

            logging.info(f"Number of unique gene embeddings: {len(gene_embedding_accumulator)}")
            logging.info("Finished calculating gene embeddings.")

            prediction["gene_embeddings"] = gene_embedding_accumulator
            prediction["gene_counts"] = input_id_count
            prediction["ensembl_IDs"] = ensembl_IDs
            prediction["gene_symbols"] = gene_symbols

        torch.save(prediction, result_path)
        if isinstance(prediction, dict):
            keys = prediction.keys()
        else:
            keys = "tensor"
        logging.info(f"Inference predictions are stored in {result_path}\n{keys}")
