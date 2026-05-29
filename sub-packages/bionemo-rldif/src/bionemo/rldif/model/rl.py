import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque, namedtuple
from copy import deepcopy
from typing import *
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import utils74 as utils
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import scipy.stats as ss
from scipy.stats import shapiro, skew
from torch.optim import AdamW
from utils.utils import get_polynomial_learning_rate

class BufferEntry:
    def __init__(
        self, *, sample, logprob_old, advantage, name, original_advantage, **kwargs
    ):
        self.sample = sample
        self.logprob_old = logprob_old
        self.advantage = advantage
        self.name = name
        self.original_advantage = original_advantage

        self.__dict__.update(kwargs)

    def _replace(self, **kw) -> "BufferEntry":
        new_dict = self.__dict__.copy()
        new_dict.update(kw)
        return BufferEntry(**new_dict)

class RLModel(nn.Module):
    @abstractmethod
    def generate_samples(
        self, batch: Any, n_samples: int, temperature: float
    ) -> List[BufferEntry]:
        # This function should take in a batch and generate n_samples samples for each
        # element of the batch. The output should be a list of BufferEntries - one per
        # generated sample.
        ...

    @abstractmethod
    def collate_fn_buffer(self, batch: List[BufferEntry]) -> Any:
        # This function should accept a set of BufferEntries and construct a batch out of
        # them. The output should be a batch that can be passed to the model for training.
        ...

    @abstractmethod
    def compute_logprobs(self, batch: Any) -> torch.FloatTensor:
        # This function should accept a batch and return the log probabilities of the
        # samples in the batch. The output should be a tensor of shape [batch_size].
        ...

    def compute_logprobs_nograd(self, batch: Any, *args, **kwargs) -> torch.FloatTensor:
        with torch.no_grad():
            ret = self.compute_logprobs(batch, *args, **kwargs)

            if torch.is_tensor(ret):
                ret = ret.detach()
            else:
                ret = [r.detach() for r in ret]

            return ret

    def ppo_step(
        self,
        batch,
        mode ,
        ppo_config,
        pi_init = None,
    ):
        model_batch = self.colocate_data(
            self.collate_fn_buffer([b.sample for b in batch])
        )

        if mode == 0:
            self.train()
            logprobs_new = self.compute_logprobs(model_batch)
        else:
            self.eval()
            logprobs_new = self.compute_logprobs_nograd(model_batch)

        logprobs_old = torch.tensor(
            [s.logprob_old for s in batch], device=logprobs_new.device
        )

        # Compute PPO objective
        ratio = torch.exp(logprobs_new - logprobs_old)
        print(f"Ratio: {ratio}", flush=True)

        adv = self.colocate_data(torch.tensor([s.advantage for s in batch]))
        print(f"Advantage: {adv}", flush=True)

        # if kl_init_weight is not None:
        #     logprobs_token_init = torch.stack([s.logprobs_token_init for s in batch], dim=0).to(logprobs_token_new.device)
        #     kl_div = torch.distributions.kl.kl_divergence(
        #         torch.distributions.Categorical(logits=logprobs_token_new),
        #         torch.distributions.Categorical(logits=logprobs_token_init),
        #     )
        #     kl_div = kl_div.sum(dim=-1)  # sum along seq dim
        #     adv = adv - (kl_div * kl_init_weight)
        #     print("KL(current, init):", kl_div)
        #     kl_div_mean = kl_div.mean()

        # else:
        #     kl_div_mean = None

        surr1 = ratio * adv
        # print(f"Surr1: {surr1}")
        surr2 = (
            torch.clamp(ratio, 1 - ppo_config.eps_clip, 1 + ppo_config.eps_clip) * adv
        )
        # print(f"Surr2: {surr2}")
        loss = -torch.min(surr1, surr2)
        # print(f"Loss: {loss}")
        clip_count = 1 / loss.shape[0] * (loss != -surr1).sum()
        print(f"Clip count:", clip_count)

        print(f"Advantage Shapiro test: {shapiro(adv.cpu().numpy())}")
        print(f"Advantage skew: {skew(adv.cpu().numpy())}")
        print(
            f"Loss skew: {torch.mean(((loss - torch.mean(loss)) /torch.std(loss)) ** 3)}"
        )
        print(f"Old logprobs: {logprobs_old.mean()}")
        print(f"New logprobs: {logprobs_new.mean()}")

        # return
        if mode == 0:
            return {
                "loss": loss.mean(),
                "advantage": adv.float().mean(),
                "advantage_std": adv.std(),
                "ratio": ratio.mean(),
                "clip_count": clip_count,
            }
        elif mode == 1:
            return {
                "loss": loss.mean(),
                "advantage": adv.float().mean(),
                "advantage_std": adv.std(),
                "ratio": ratio.mean(),
                "clip_count": clip_count,
            }

    def ppo_validation_step(self, batch, *args, **kwargs):
        return self.ppo_step(batch, *args, mode=1, **kwargs)


class RLApproach(pl.LightningModule):
    def __init__(
        self,
        config,
        model: RLModel,
        dataloaders: Optional[
            Union[Dict[str, DataLoader[Any]], Callable[[str], DataLoader[Any]]]
        ],
    ) -> None:
        
        super().__init__()

        self.config = config
        # assert isinstance(model, RLXFModel), "model must inherit from RLXFModel"

        self.models = {"validation_model": model}

        if dataloaders is None:
            logger.info(
                "No dataloaders found - approach configured for inference mode only"
            )
        elif isinstance(dataloaders, Callable):
            self.dataloader_func = dataloaders
        else:
            self.dataloaders = dataloaders

        self.model = model
        self.automatic_optimization = False

        self.model_init = None

        self._adv_stats = None
        self._episode_counter = 0
        self._update_batch_size = 8
        self._stat_tracker = PerProteinStatTracker(
            buffer_size= config.data.n_samples
        )
        self.train_config = config

    @property
    def pi(self) -> RLModel:
        return self.model

    @property
    def pi_init(self) -> RLModel:
        return self.model_init

    def _set_adv_stats(self, mean: float, std: float):
        self._adv_stats = (mean, std)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        if self._adv_stats is not None:
            mean_, std_ = self._adv_stats
            # for logger in self.loggers:
            self.log(name="Mean Train Advantage", value=mean_, batch_size=1)
            self.log(name="neg_train_advantage", value=-mean_, batch_size=1)
            self.log(name="STD Train Advantage", value=std_, batch_size=1)
            self._adv_stats = None

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        with torch.no_grad():
            self.pi.eval()

            gen_kw = dict(random_mutation_rate=self.config.random_mutation_rate)
            # if self.config.ppo.lambda_kl_init is not None:
            #     gen_kw.update(pi_init=self.pi_init)

            buffer = self.pi.generate_samples(
                batch, self.config.data.n_samples, **gen_kw
            )

        buffer = self.prep_buffer(buffer)

        opt = self.optimizers()
        losses = []
        advantages = []
        ratios = []
        clip_counts = []
        kls = []

        for i_inner in range(self.config.n_update_epochs):
            print(f"Inner epoch: {i_inner}")
            batch_count = 0
            for batch in self.get_batches(buffer, first=i_inner == 0):
                self.pi.train()
                train_dict = self.pi.ppo_step(
                    batch,
                    mode=0,
                    ppo_config=self.config.ppo,
                    pi_init=self.pi_init,
                )

                loss = train_dict["loss"] / self.config.data.n_sampling_batches_per_step
                self.manual_backward(loss)

                if (batch_count + 1) % self.config.opt.gradient_accumulation_steps == 0:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.config.opt.clip_grad_norm,
                        gradient_clip_algorithm="norm",
                    )
                    opt.step()
                    opt.zero_grad()

                    sch = self.lr_schedulers()
                    sch.step()

                losses.append(loss.item())
                advantages.append(train_dict["advantage"].item())
                ratios.append(train_dict["ratio"].item())
                clip_counts.append(train_dict["clip_count"].item())
                # if train_dict["kl"] is not None:
                #     kls.append(train_dict["kl"].item())
                batch_count += 1

            print()

        self.log(
            name="mean_train_step_loss", value=torch.tensor(losses).mean(), batch_size=1
        )
        self.log(
            name="mean_train_step_ratio",
            value=torch.tensor(ratios).mean(),
            batch_size=1,
        )
        self.log(
            name="mean_train_step_clip",
            value=torch.tensor(clip_counts).mean(),
            batch_size=1,
        )
        # if kls:
        #     self.log(name="mean_train_step_kl", value=torch.tensor(kls).mean(), batch_size=1)

        return torch.tensor(losses).sum()

    def train_dataloader(self):
        # Don't want to call set_batch_size here because the sampling batch size of the dataloader is not the one
        # we typically want to tune.
        if not hasattr(self, "dataloaders"):
            self.dataloaders = {"train": self.dataloader_func("train")}
        dl = self.dataloaders["train"]
        return dl

    @property
    def batch_size(self) -> int:
        return self._update_batch_size

    @batch_size.setter
    def batch_size(self, value):
        logger.info(f"Updating batch size to {value}")
        self._update_batch_size = value

    def prep_buffer(self, buffer):
        # normalize advantages to have mean 0 and std 1

        # advantages = {}
        seqs = defaultdict(list)
        accs = []
        for s in buffer:
            protein_id = s.name
            # mask = s.sample["mask"].bool()
            # seq = s.sample["S"][mask]
            # seqs[protein_id].append(seq)

            self._stat_tracker.add_rewards(protein_id, [s.advantage])
            # true_seq = s.sample["True_Sequences"][mask]
            # acc = (seq == true_seq).float().mean().item()
            accs.append(s.advantage)

        accs = torch.tensor(accs)
        self.log(name="mean_train_accuracy", value=accs.mean(), batch_size=1)
        logger.info(
            f"Accuracy (ep={self._episode_counter}): {accs.mean().item():.3g} +/- {accs.std().item():.3g}"
        )

        all_stats = self._stat_tracker.get_all_stats()
        all_means, all_stds = [], []
        divs = []
        for k in sorted(all_stats.keys()):
            if k not in seqs:
                continue

            st = all_stats[k]
            mean, std = st["mean"], st["std"]
            logger.info(
                f'Advantage (ep={self._episode_counter}) Seq {k}: {mean:.3g} +/- {std:.3g} (n={st["count"]})'
            )

            all_means.append(mean)
            all_stds.append(std)

            div = diversity(seqs[k])
            logger.info(f"Diversity (ep={self._episode_counter}) Seq {k}: {div:.3g}")
            divs.append(div)

        self.log(
            name="mean_train_diversity", value=torch.tensor(divs).mean(), batch_size=1
        )

        mean_, std_ = np.mean(all_means), np.mean(all_stds)
        logger.info(
            f"Advantage (ep={self._episode_counter}): {mean_:.3g} +/- {std_:.3g}"
        )

        total_advs = torch.tensor([s.advantage for s in buffer])
        mean, std = total_advs.mean(), total_advs.std()
        mean_, std_ = mean.item(), std.item()
        self.log(name="Mean Train Advantage", value=mean_, batch_size=1)
        self.log(name="neg_train_advantage", value=-mean_, batch_size=1)
        self.log(name="STD Train Advantage", value=std_, batch_size=1)

        # this is the spread across proteins
        self.log(
            name="STD(Mean Train Advantage)", value=np.std(all_means), batch_size=1
        )
        self.log(name="STD(STD Train Advantage)", value=np.std(all_stds), batch_size=1)
        self.log(
            name="Mean(STD Train Advantage)", value=np.mean(all_stds), batch_size=1
        )

        rank_total_advs = total_advs.tolist()
        rank_total_advs.sort()
        for i in range(len(buffer)):
            st = all_stats[buffer[i].name]
            mean, std = st["mean"], self.config.fixed_advantage_scale or st["std"]
            if self.train_config.log_normalize:
                buffer[i].advantage = self._stat_tracker.get_log_normalize_stats(
                    buffer[i].name
                )[buffer[i].advantage]
            elif self.train_config.ranking:
                buffer[i].advantage = self._stat_tracker.get_ranked_stats(
                    buffer[i].name
                )[buffer[i].advantage]
            elif self.train_config.box_cox:
                buffer[i].advantage = self._stat_tracker.get_box_cox_stats(
                    buffer[i].name
                )[buffer[i].advantage]
            else:
                buffer[i].advantage = (buffer[i].advantage - mean) / (std + 1e-5)

        self._episode_counter += 1
        return buffer

    def get_batches(self, buffer, first: bool):
        # construct batches from the buffer

        if not first and self.config.update_old_logprobs:
            # update logprobs_old - hacky but not horrible since inference is the bottleneck
            logger.info("Updating old logprobs!")
            for i in range(len(buffer)):
                sample = self.pi.collate_fn_buffer([buffer[i].sample])
                sample = self.pi.colocate_data(sample)
                buffer[i] = buffer[i]._replace(
                    logprob_old=self.pi.compute_logprobs_nograd(sample).cpu()[0].item()
                )

        idcs = np.random.permutation(len(buffer))
        batch_idcs = utils.data.make_chain(idcs, size=self.batch_size)
        for batch in batch_idcs:
            yield [buffer[i] for i in batch]

    def _get_n_ppo_steps(self) -> int:
        n_ppo_steps = math.ceil(
            self.config.n_update_epochs
            * self.config.data.sampling_batch_size_per_gpu
            * self.config.data.n_samples
            / self.config.data.update_batch_size_per_gpu
        )
        return n_ppo_steps

    def configure_optimizers(self) -> Dict[str, Union["Optimizer", Dict[str, Any]]]:
        # override this to set spe correctly

        if not hasattr(self, "dataloaders"):
            self.dataloaders = {"train": self.dataloader_func("train", self)}
        
        optimizer = AdamW(self.model.parameters(), lr = self.config.lr.lr)
        self.model.opt = optimizer

        step_scale_factor = 1
        
        if self.config.n_epochs is not None:
            step_scale_factor *= self.config.n_epochs
        self.config.n_update_epochs = 1
        step_scale_factor /= self.config.opt.gradient_accumulation_steps
        step_scale_factor /= self.trainer.num_nodes * self.trainer.num_devices
        step_scale_factor *= self._get_n_ppo_steps()

        scheduler = self.get_lr(
            optimizer,
            train_dataloader=self.dataloaders["train"],
            step_scale_factor=step_scale_factor,
            config=self.config,
        )
        self.model.lr_sched = scheduler

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_lr(self, optimizer, train_dataloader, step_scale_factor, config):
        return get_polynomial_learning_rate(
                optimizer,
                num_warmup_steps=config.lr.warmup_steps,
                num_training_steps=100000,
                train_dataloader=self.dataloaders["train"],
                step_scale_factor=step_scale_factor,
                power=config.lr.gamma,
            )


    def set_training_pbar(self, trainer) -> utils.lqdm:
        lqdm_kwargs: Dict[str, Any] = {"desc": f"Epoch {trainer.current_epoch}: TRAIN"}
        ppo_steps = self._get_n_ppo_steps()

        if self.config.n_train_steps is not None:
            lqdm_kwargs["total"] = self.config.n_train_steps * ppo_steps
            lqdm_kwargs["initial"] = trainer.global_step

        elif self.config.epoch_train_steps is not None:
            lqdm_kwargs["total"] = self.config.epoch_train_steps * ppo_steps

        else:
            lqdm_kwargs["total"] = (
                len(self.train_dataloader())
                * ppo_steps
                # // self.config.opt.gradient_accumulation_steps
                // (trainer.num_nodes * trainer.num_devices)
            )
        self.pbar = self.lqdm(**lqdm_kwargs)


# Helper for utility function
def utility_fn(adv: torch.FloatTensor) -> torch.FloatTensor:
    return adv

class PerProteinStatTracker:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.stats = defaultdict(lambda: deque(maxlen=self.buffer_size))

    def add_rewards(self, protein_id, rewards):
        self.stats[protein_id].extend(rewards)

    def get_stats(self, protein_id):
        mean = np.mean(self.stats[protein_id])
        std = np.std(self.stats[protein_id]) + 1e-6
        return mean, std

    def get_ranked_stats(self, protein_id):
        rewards = self.stats[protein_id]
        ranked_rewards = ss.rankdata(rewards)
        ranked_reward_mapping = {}
        for i, z in zip(rewards, ranked_rewards):
            if np.std(ranked_rewards) == 0:
                ranked_reward_mapping[i] = z - np.mean(ranked_rewards)
            else:
                ranked_reward_mapping[i] = (z - np.mean(ranked_rewards)) / np.std(
                    ranked_rewards
                )

        return ranked_reward_mapping

    def get_log_normalize_stats(self, protein_id):
        rewards = self.stats[protein_id]
        sub_rewards = []
        for i in rewards:
            if not i:
                sub_rewards.append(0.0001)
            else:
                sub_rewards.append(i)

        log_rewards = np.log(sub_rewards)
        reward_mapping = {}
        for i, z in zip(rewards, log_rewards):
            if np.std(log_rewards) == 0:
                reward_mapping[i] = z - np.mean(log_rewards)
            else:
                reward_mapping[i] = (z - np.mean(log_rewards)) / np.std(log_rewards)

        return reward_mapping

    def get_box_cox_stats(self, protein_id):
        rewards = self.stats[protein_id]
        sub_rewards = []
        if len(set([i.item() for i in rewards])) == 1:
            sub_rewards = [i + (np.random.random() / 1000) for i in rewards]
        else:
            sub_rewards = rewards

        final_rewards = []
        for i in sub_rewards:
            if not i.item():
                final_rewards.append(0.0001)
            else:
                final_rewards.append(i)

        print(rewards)
        print(final_rewards)
        box_rewards = ss.boxcox(final_rewards / np.mean(final_rewards))[0]
        print(box_rewards)
        # import pdb

        # pdb.set_trace()
        reward_mapping = {}
        for i, z in zip(rewards, box_rewards):
            reward_mapping[i] = (z - np.mean(box_rewards)) / np.std(box_rewards)

        return reward_mapping

    def get_all_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v) + 1e-6, "count": len(v)}
            for k, v in self.stats.items()
        }
