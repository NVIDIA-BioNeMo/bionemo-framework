import torch
from typing import *
import math
import torch.nn.functional as F
from torch import nn
from ai.tools.dl import t2n
from loguru import logger
from torch_geometric.data import Batch as GraphBatch
import numpy as np
from typing import *
from bionemo.rldif.utils.utils import LongType, FloatTensor, LongTensor
from tqdm import tqdm

class BaseDiffuser:
    def __init__(self, config):
        super().__init__() 
        self.T = config.T
        self.config = config
        self.timesteps = torch.arange(self.T, 0, -config.t_delta, dtype=torch.long)
        self._ext_timesteps = torch.cat([self.timesteps, torch.LongTensor([0])])
        self.set_schedule(config)

    def beta(self, t: LongType, *, like: FloatTensor = None, like_shape: Tuple = None):
        return self._extract(self._beta, t, like=like, like_shape=like_shape)

    def alpha(self, t: LongType, *, like: FloatTensor = None, like_shape: Tuple = None):
        return self._extract(self._alpha, t, like=like, like_shape=like_shape)

    def alphabar(
        self, t: LongType, *, like: FloatTensor = None, like_shape: Tuple = None
    ):
        return self._extract(self._alphabar, t, like=like, like_shape=like_shape)

    def alphabar_prev(
        self, t: LongType, *, like: FloatTensor = None, like_shape: Tuple = None
    ):
        prev_t = self.get_prev_timestep(t)
        kw = dict(like=like, like_shape=like_shape)
        if torch.is_tensor(prev_t):
            prev_t_clamped = torch.clamp(prev_t, min=1)
            ab = self.alphabar(prev_t_clamped, **kw)
            return torch.where(
                self._make_like(prev_t >= 1, ab),
                ab,
                torch.tensor(1.0, device=ab.device),
            )
        else:
            if like_shape is None:
                like_shape = like.shape
            return (
                self.alphabar(prev_t, **kw)
                if prev_t >= 1
                else self._make_like(torch.ones(like_shape[0]), like)
            )

    def posterior_variance(
        self, t: LongType, *, like: FloatTensor = None, like_shape: Tuple = None
    ):
        return self._extract(
            self._posterior_variance, t, like=like, like_shape=like_shape
        )

    def _extract(
        self,
        param: FloatTensor,
        t: LongType,
        *,
        like: FloatTensor = None,
        like_shape: Tuple = None,
    ):
        if like is not None:
            like_shape = like.shape

        t_idx = self.get_timestep_idx(t)
        if isinstance(t_idx, int):
            assert t_idx >= 0, f"Trying to access noise at t={t}."
            out = torch.ones(size=(like_shape[0],)) * param[t_idx]
        else:
            assert t_idx.min() >= 0, f"Trying to access noise at t={t.min().item()}."
            out = torch.gather(param, 0, t_idx.to(param.device))

        if like is None:
            return out
        else:
            reshape = [like_shape[0]] + [1] * (len(like_shape) - 1)
            return out.reshape(*reshape).to(device=like.device)

    @staticmethod
    def is_zero(t: LongType):
        if isinstance(t, torch.Tensor):
            return t.abs().sum() == 0
        else:
            return t == 0

    @staticmethod
    def _mask_values(
        x: FloatTensor, *, ref: FloatTensor, mask: FloatTensor
    ) -> FloatTensor:
        if mask is None:
            return x

        fmask = mask.float()
        if fmask.shape != x.shape:
            nd_extra = len(x.shape) - len(fmask.shape)
            fmask = fmask.reshape(tuple(fmask.shape) + ((1,) * nd_extra))

        x = (fmask * x) + ((1 - fmask) * ref)

        return x

    @staticmethod
    def _mask_eps(eps: FloatTensor, mask: FloatTensor) -> FloatTensor:
        if mask is None:
            return eps

        bmask = mask.bool()
        eps[~bmask] = 0

        return eps

    def posterior_variance_unused(
        self, t: LongType, *, like: FloatTensor = None, like_shape: Tuple = None
    ):
        # calculate on the fly, but now we calculate on init
        if like is not None:
            like_shape = like.shape

        if isinstance(t, int):
            t = torch.ones(like_shape[:1], dtype=torch.long) * t
        t_shape = t.shape[:1]

        beta = self.beta(t, like_shape=t_shape)
        ab = self.alphabar(t, like_shape=t_shape)
        t_prev = self.get_prev_timestep(t)
        t_prev_clipped = torch.clamp(t_prev, min=1)

        ab_prev = torch.where(
            t_prev >= 1,
            self.alphabar(t_prev_clipped, like_shape=t_shape),
            torch.tensor(0.9999),
        )

        out = beta * (1.0 - ab_prev) / (1.0 - ab)

        if like_shape is None:
            return out
        else:
            reshape = [like_shape[0]] + [1] * (len(like_shape) - 1)
            return out.reshape(*reshape).to(device=like.device)

    def set_schedule(self, config):
        # sched = config.noise_schedule
        T = config.T // config.t_delta
        kwargs = {}
        #kwargs = config.scheduler_args

        def from_beta(beta):
            alpha = 1 - beta
            alphabar = torch.cumprod(alpha, axis=0)
            alphabar_sqrt = torch.sqrt(alphabar)

            # from https://arxiv.org/pdf/2305.08891.pdf
            # Store old values.
            alphabar_sqrt_0 = alphabar_sqrt[0].clone()
            alphabar_sqrt_T = alphabar_sqrt[-1].clone()
            # Shift so last timestep is zero.
            alphabar_sqrt -= alphabar_sqrt_T
            # Scale so first timestep is back to old value.
            alphabar_sqrt *= alphabar_sqrt_0 / (alphabar_sqrt_0 - alphabar_sqrt_T)

            # Convert alphas_bar_sqrt to betas
            alphabar = alphabar_sqrt**2
            alpha = alphabar[1:] / alphabar[:-1]
            alpha = torch.cat([alphabar[0:1], alpha])
            beta = 1 - alpha

            return beta, alpha, alphabar

        s = kwargs.get("s", 8e-3)
        t = torch.linspace(0, 1, T + 1)  # need T+1 b/c alphabar goes from 0...T
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        f[T] = 0

        alphabar = f / f[0]

        alpha = alphabar[1:] / alphabar[:-1]  # 1...T
        alpha = torch.cat([alphabar[0:1], alpha])
        beta = 1 - alpha


        assert (
            alphabar[T] == 0.0
        ), "alphabar_T should be 0 - c.f. https://arxiv.org/pdf/2305.08891.pdf"
        assert (
            beta[T] == 1.0
        ), "beta_T should be 1 - c.f. https://arxiv.org/pdf/2305.08891.pdf"

        # alphabar_shifted = F.pad(alphabar[:-1], (1, 0), value=0.9999)
        # posterior_variance_old = beta * (1.0 - alphabar_shifted) / (1.0 - alphabar)

        # variance of posterior q(x_{t-1} | x_t)
        posterior_variance = beta[1:] * (1.0 - alphabar[:-1]) / (1.0 - alphabar[1:])
        # should never be accessing t=0, since that's q(x_-1 | x_0)
        # just pad so things are the right dim
        posterior_variance = torch.cat([torch.tensor([0.0]), posterior_variance])

        self._beta = beta.float()
        self._alpha = alpha.float()
        self._alphabar = alphabar.float()
        self._posterior_variance = posterior_variance.float()

    @staticmethod
    def _make_like(t, other):
        shape = other.shape
        if not torch.is_tensor(t):
            t = torch.FloatTensor([t])
        assert t.ndim <= len(shape)
        return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).to(other.device)

    def get_prev_timestep(self, t: LongType) -> LongType:
        """
        Get the value(s) immediately to the right of `t` in `timesteps`.

        Args:
            timesteps (LongTensor): 1D tensor of timesteps.
            t (int or LongTensor): Value(s) to search for in `timesteps`.

        Returns:
            int or LongTensor: Value(s) immediately to the right of `t` in `timesteps`.

        Raises:
            ValueError: If `t` is not in `timesteps`.
        """
        if single := isinstance(t, int):
            t = torch.tensor([t], dtype=torch.long, device=self._ext_timesteps.device)
        else:
            t = t.long().to(self._ext_timesteps.device)

        idx = (self._ext_timesteps == t.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        if idx.numel() == 0:
            raise ValueError(f"t={t} not found in _ext_timesteps")
        idx += 1
        if idx.max() >= self._ext_timesteps.shape[0]:
            raise ValueError(f"t={t} is the first timestep in _ext_timesteps")

        ret = self._ext_timesteps[idx.flatten()]
        if single:
            return ret.item()
        else:
            return ret.to(t.device)

    def get_timestep_idx(self, t: LongType) -> LongType:
        t_idx = t // self.config.t_delta
        return t_idx

class CategoricalBaseDiffusionConfig():
    num_categories: int = 21  # Needed for noise calculation


class CategoricalBaseDiffuser(BaseDiffuser):
    # CODE inspired from https://arxiv.org/pdf/2102.05379.pdf
    # We used the pseudocode from here

    def __init__(self, config: CategoricalBaseDiffusionConfig):
        self.config = config
        super().__init__(config)

    def log_add_exp(self, a, b):
        maximum = torch.max(a, b)
        return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

    def log_sum_exp(self, x):
        maximum = torch.max(x, dim=1, keepdim=True).values
        return maximum + torch.log(torch.exp(x - maximum).sum(dim=1, keepdim=True))

    def log_1_min_a(self, a):
        return torch.log(1 - a.exp() + 1e-30)

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha = self.alpha(t, like=log_x_t).log()
        log_1_min_alpha = self.log_1_min_a(log_alpha)
        log_probs = self.log_add_exp(
            log_x_t + log_alpha, log_1_min_alpha - np.log(self.config.num_categories)
        )
        return log_probs

    def q_pred(self, log_x0, t):
        log_alphabar = self.alphabar(t, like=log_x0).log()
        log_1_min_alphabar = self.log_1_min_a(log_alphabar)
        log_probs = self.log_add_exp(
            log_x0 + log_alphabar,
            log_1_min_alphabar - np.log(self.config.num_categories),
        )
        return log_probs

    def q_posterior(self, log_x0, log_x_t, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_probs_xtmin = self.q_pred(log_x0, t_minus_1)
        num_axes = (1,) * (len(log_x0.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x0)
        log_probs_xtmin = torch.where(t_broadcast == 0, log_x0, log_probs_xtmin)

        unnormed_logprobs = log_probs_xtmin + self.q_pred_one_timestep(log_x_t, t)
        log_probs_posterior = unnormed_logprobs - torch.logsumexp(
            unnormed_logprobs, dim=1, keepdim=True
        )
        return log_probs_posterior

    def noise_features(
        self, t: LongTensor, f_0: FloatTensor, noise_mask: FloatTensor = None
    ) -> Tuple[FloatTensor, FloatTensor]:
        if self.is_zero(t):
            return f_0

        # Features are assumed to be always one-hot encoded AA, based on distribution
        # Noise the initial distribution, then sample from it to create our new features
        f_t = self.q_pred(torch.log(f_0.clamp(min=1e-30)), t)
        f_t = self.log_sample_categorical(f_t)
        f_t = self._mask_values(f_t, ref=f_0, mask=noise_mask).float()
        return f_t

    def p_pred(
        self,
        t: LongType,
        f_t: FloatTensor,
        eps_pred: FloatTensor,
        noise_mask: FloatTensor = None,
    ) -> FloatTensor:
        log_x_recon = torch.nn.LogSoftmax(dim=1)(eps_pred)
        log_model_pred = self.q_posterior(
            log_x_recon, torch.log(f_t.clamp(min=1e-30)), t
        )

        return log_model_pred.exp()

    def log_sample_categorical(self, logits):
        # Paper does this, adds some gumbel noise when sampling
        # So we just don't sample the same thing every time

        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        sample = F.one_hot(sample, self.config.num_categories)
        return sample

    def log_sample_categorical_with_logprobs(self, sample, logits):
        # Gather the log probabilities of the samples
        return logits.masked_select(sample.bool())

    def denoise_feature_eps(
        self,
        t: LongType,
        f_t: FloatTensor,
        eps_pred: FloatTensor,
        noise_mask: FloatTensor = None,
    ) -> FloatTensor:
        # eps_pred in this case is our model's prediction of x_0
        logits = self.p_pred(t, f_t, eps_pred, noise_mask)
        sample = self.log_sample_categorical(logits.log())
        # Use this to then predict AA distribution

        return sample

    def denoise_feature_eps_with_logprobs(
        self,
        t: LongType,
        f_t: FloatTensor,
        eps_pred: FloatTensor,
        noise_mask: FloatTensor = None,
    ) -> FloatTensor:
        # eps_pred in this case is our model's prediction of x_0
        logits = self.p_pred(t, f_t, eps_pred, noise_mask)
        sample = self.log_sample_categorical(logits.log())
        return sample, self.log_sample_categorical_with_logprobs(sample, logits.log())

    def categorical_kl(self, pred: FloatTensor, target: FloatTensor) -> FloatTensor:
        log_prob_a = torch.log(pred)
        log_prob_b = torch.log(target)
        kl = (log_prob_a.exp() * (log_prob_a - log_prob_b)).sum(dim=1)
        # import pdb

        # pdb.set_trace()
        return kl

    def log_categorical(self, log_x_start, log_prob):
        return (log_x_start.exp() * log_prob).sum(dim=1)

    def compute_losses(
        self, batch: GraphBatch, mode
    ) -> GraphBatch:
        # computes and stores losses in-place
        log_x_start = torch.log(batch.features_0.clamp(min=1e-30))

        # Batch.features_pred is raw output have to take the softmax of it
        log_true_prob = self.q_posterior(
            log_x_start,
            torch.log(batch.features_t.clamp(min=1e-30)),
            batch.t,
        ).exp()

        log_model_prob = self.p_pred(
            batch.t, batch.features_t, batch.features_pred, batch.mask
        )

        # Take average over every position
        kl = self.categorical_kl(log_true_prob, log_model_prob)

        # For if t is 0
        decoder_nll = -self.log_categorical(log_x_start, log_model_prob)
        mask = (batch.t == torch.zeros_like(batch.t)).float()
        kl = mask * decoder_nll + (1.0 - mask) * kl
        kl = kl[batch.mask.bool()]

        kl_prior = self.kl_prior(log_x_start)[batch.mask.bool()]

        pt = torch.ones_like(batch.t).float() / self.T
        loss = kl / pt[batch.mask.bool()] + kl_prior
        loss = loss.sum() / (math.log(2) * len(loss))
        batch.loss = loss


        return batch

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()
        log_qxT_prob = self.q_pred(log_x_start, t=(self.T - 1) * ones)
        log_half_prob = -torch.log(
            self.config.num_categories * torch.ones_like(log_qxT_prob)
        )
        kl_prior = self.categorical_kl(log_qxT_prob.exp(), log_half_prob.exp())
        return kl_prior

    def categorical_kl_loss_graph_mean(
        self,
        input: FloatTensor,
        target: FloatTensor,
        batch: torch.LongTensor,
        mask: FloatTensor = None,
    ) -> FloatTensor:
        # like mse_loss, but we first average over nodes in each graph, then average over all graphs. this is what self.mse_loss() does if a loss_mask
        # is provided (assuming the loss_mask includes a pad mask).

        # Input is model prob
        # Target is true prob

        bsz = batch.max() + 1
        n = input.shape[0]
        nd = len(input.shape)

        if mask is None:
            mask = input.new_ones((n,))
        mask = mask.float()
       
        # Should be TRUE, MODEL PROB
        err = self.categorical_kl(target, input)
        err = err * mask
        loss_num = err.new_zeros((bsz,))
        loss_num.index_add_(0, batch, err)

        return loss_num
