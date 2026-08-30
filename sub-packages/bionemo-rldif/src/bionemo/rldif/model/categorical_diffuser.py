from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch as GraphBatch, Data as GraphData
import torch_cluster
import numpy as np
import math
from einops import rearrange
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
from bionemo.rldif.utils.utils import t2n
from bionemo.rldif.model.diffuser import CategoricalBaseDiffuser, CategoricalBaseDiffusionConfig
from bionemo.rldif.utils.utils import t2n, TRAIN, VALIDATION, INFERENCE, SAMPLING

class CategoricalDiffusionConfig(CategoricalBaseDiffusionConfig):
    name: str
    scale_features: bool = (
        False  # if True, scale the features instead of scaling the feature loss
    )
    feature_scale: float = 0.25
    use_pifold: bool = False
    use_gradeIF: bool = False

class CategoricalDiffuser(CategoricalBaseDiffuser):

    def __init__(self, config):
        self.config = config
        super().__init__(config)
    
    def colocate_data(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch

    def init_sample_batch(self, batch: GraphBatch) -> GraphBatch:
        # in this case, we use batch as a template: number of samples, size of each graph, etc
        batch = self.colocate_data(batch)
        batch.t = (
            torch.ones(
                (batch.batch.shape[0],), device=batch.batch.device, dtype=torch.float32
            )
            * self.T
        )
        
        uniform_logits = torch.zeros(
            (batch.features_0.shape[0], self.config.num_categories),
            device=batch.batch.device,
        )
        batch.features_t = self.log_sample_categorical(uniform_logits)

        return batch

    @torch.no_grad()
    def denoise_batch(self, batch: GraphBatch, t: int) -> GraphBatch:
        if isinstance(t, int):
            t = (
                torch.ones(
                    batch.features_t.shape[:1],
                    device=batch.features_t.device,
                    dtype=torch.int64,
                )
                * t
            )
        batch.t = t.float()

        output = self.sample_step(batch)

        output.features_step = self.denoise_feature_eps(
            t=t,
            f_t=batch.features_t,
            eps_pred=output.features_pred,
        )

        return output

    def denoise_batch_with_logprobs(self, batch: GraphBatch, t: int) -> GraphBatch:
        if isinstance(t, int):
            t = (
                torch.ones(
                    batch.features_t.shape[:1],
                    device=batch.features_t.device,
                    dtype=torch.int64,
                )
                * t
            )
        batch.t = t.float()

        output = self.sample_step(batch)

        (
            output.features_step,
            output.features_logprobs,
        ) = self.denoise_feature_eps_with_logprobs(
            t=t,
            f_t=batch.features_t,
            eps_pred=output.features_pred,
        )

        return output

    def noise_batch(
        self, batch: GraphBatch, t: int = None, t_max: int = None
    ) -> GraphBatch:
        # noise a batch graph
        batch = self.colocate_data(batch)

        b = batch.batch.max() + 1

        if t is None:
            t = torch.randint(1, t_max or (self.T + 1), (b,), device=batch.batch.device)
        elif isinstance(t, int):
            t = torch.ones((b,), device=batch.batch.device, dtype=torch.int64) * t

        # broadcast across nodes
        t = t[batch.batch]
        batch.t = t
        batch.features_t = self.noise_features(t=t, f_0=batch.features_0)
        return batch

    @torch.no_grad()
    def sample(
        self,
        batch: GraphBatch,
        t_start: int = None,
        loud: bool = False,
        closure: bool = False,
    ) -> dict:
        if t_start is not None:
            logger.warning(
                f"Starting sampling at t={t_start}. Should only be used for debugging."
            )

        batch_idcs = batch.batch
        n_batches = t2n(batch_idcs.max() + 1)

        def get(attr, b, scale=1, mask=False):
            if mask:
                return [
                    t2n(
                        getattr(b, attr)[batch_idcs.cuda()[batch.mask.bool()] == i_b]
                        / scale
                    ).copy()
                    for i_b in range(n_batches)
                ]
            else:
                return [
                    t2n(getattr(b, attr)[batch_idcs == i_b] / scale).copy()
                    for i_b in range(n_batches)
                ]

        buffer = {}

        ft_scale = 1
        buffer["features_true"] = get("features_0", batch, ft_scale)
        buffer["mask"] = get("mask", batch)

        t_start = t_start or self.T

        batch = self.init_sample_batch(batch)

        # store the initial noised features as the first step
        buffer[f"features_{t_start}_step"] = get("features_t", batch, ft_scale)

        for t in range(t_start, 0, -1):
            # each step of this loop takes in f_t and estimates f_t-1
            if t != t_start:
                # not the first step, so update f_t from the previous step
                batch.features_t = batch.features_step.clone()

            batch = self.denoise_batch(batch, t)

            tm1 = t - 1
            if tm1 % 1 == 0:
                buffer[f"features_{tm1}_step"] = get("features_step", batch, ft_scale)
                
        return buffer

    @torch.no_grad()
    def sample_step(self, batch: GraphBatch) -> GraphBatch:
        self.eval()

        return self.step(batch, SAMPLING)
