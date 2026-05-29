from copy import deepcopy
import math
from collections import defaultdict, deque
from pprint import pprint
from typing import *

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader


from model.rl import BufferEntry, RLModel, RLApproach
import random

class RLApproachDiffusion(RLApproach):
    def __init__(
        self,
        config,
        model: RLModel,
        dataloaders: Optional[
            Union[Dict[str, DataLoader[Any]], Callable[[str], DataLoader[Any]]]
        ],
    ) -> None:
        super().__init__(config, model, dataloaders)

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

        self._episode_counter = 0
        self._update_batch_size = 8


        config.data.adv_buffer_size = config.data.n_samples

        if config.data.adv_buffer_size is not None:
            self._stat_tracker = PerSampleStatTracker(
                buffer_size=config.data.adv_buffer_size, n_advs=1
            )
        else:
            self._stat_tracker = None

        
        self.n_advs = 1
        self.config.n_diffusion_steps = 0

    @property
    def pi(self) -> RLModel:
        return self.model

    def training_step_ppo(
        self, batch: Any, batch_idx: int, is_update_step: bool = True
    ):
        with torch.no_grad():
            self.pi.eval()
            buffer = self.pi.generate_samples(batch, self.config.data.n_samples)

        buffer = self.prep_buffer(buffer)

        opt = self.optimizers()
        losses = []
        advantages = []
        original_advantages = []
        ratios = []
        clip_counts = []
        pbar = None

        for _ in range(self.config.n_update_epochs):
            batches = list(self.get_batches(buffer))
            if pbar is None:
                pbar = tqdm(
                    total=len(batches) * self.config.n_update_epochs,
                    desc="PPO steps",
                    mininterval=5,
                    disable=(self.config.verbosity < 15),
                )

            for update_weights, batch in self.get_batches(buffer):
                self.pi.train()

                if (
                    self.config.adv_normalization
                    == AdvantageNormalization.per_minibatch
                ):
                    advs = torch.tensor([s.original_advantage for s in batch])
                    mean, std = (
                        advs.mean(dim=1),
                        advs.std(dim=1) + self.config.adv_std_eps,
                    )
                    for s in batch:
                        s.advantage = (s.original_advantage - mean) / std

                for s in batch:
                    s.advantage = (s.advantage * self.adv_weights).sum()

                # This seems to be a bug, neither have config.ppo
                train_dict = self.pi.ppo_step(
                    batch,
                    mode=0,
                    eps_clip=0.03,
                    # ppo_config=self.config.ppo,
                    # pi_init=self.pi_init,
                )

                loss = train_dict["loss"] / self.config.data.n_sampling_batches_per_step

                self.manual_backward(loss)

                if is_update_step and update_weights:
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
                original_advantages.append(train_dict["original_advantage"].item())
                advantages.append(train_dict["advantage"].item())
                ratios.append(train_dict["ratio"].item())
                clip_counts.append(train_dict["clip_count"].item())

                pbar.update(1)

        pbar.close()

        self.log(
            name="mean_train_step_loss_ppo",
            value=torch.tensor(losses).mean(),
            batch_size=1,
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

        self.log(
            name="mean_train_step_advantage",
            value=torch.tensor(original_advantages).mean(),
            batch_size=1,
        )

        self._episode_counter += 1

        return torch.tensor(losses).sum()

    def training_step_diffusion(
        self, batch: Any, batch_idx: int, is_update_step: bool = True
    ):
        loss = self.pi.training_step(batch)
        self.manual_backward(loss)

        if is_update_step:
            opt = self.optimizers()
            self.clip_gradients(
                opt,
                gradient_clip_val=self.config.opt.clip_grad_norm,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            opt.zero_grad()

            sch = self.lr_schedulers()
            sch.step()

        loss = loss.detach().cpu()
        self.log(name="mean_train_step_loss_diffusion", value=loss, batch_size=1)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        if self.config.n_diffusion_steps == 0:
            # in this case just run ppo. always update
            return self.training_step_ppo(batch, batch_idx, is_update_step=True)

        elif batch_idx % (self.config.n_diffusion_steps + 1) == 0:
            # if we're running diffusion, run the ppo step first. don't update - let grads accumulate
            return self.training_step_ppo(batch, batch_idx, is_update_step=False)

        else:
            # after the first (ppo) step, run diffusion. only update weights on the last diffusion step
            is_last_diffusion = (batch_idx + 1) % (
                self.config.n_diffusion_steps + 1
            ) == 0
            return self.training_step_diffusion(
                batch, batch_idx, is_update_step=is_last_diffusion
            )

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

    def prep_buffer(self, buffer: Dict) -> List[BufferEntry]:
        trajs, advs = buffer["trajectories"], buffer["advantages"]

        # first pass through advs to fill buffer
        if self._stat_tracker is not None:
            for sample_id, sample_advs in advs.items():
                if self.config.adv_normalization == AdvantageNormalization.globally:
                    sample_id = "global"
                self._stat_tracker.add_rewards(sample_id, sample_advs)

            if self.config.adv_normalization == AdvantageNormalization.globally:
                pprint(self._stat_tracker.get_all_stats())

        # normalize per sample/globally
        for sample_id, sample_advs in advs.items():
            if self._stat_tracker is not None:
                adv_stats = self._stat_tracker.get_stats(
                    "global"
                    if self.config.adv_normalization == AdvantageNormalization.globally
                    else sample_id
                )
                # std = std + self.config.adv_std_eps
            else:
                # mean, std = 0, 1
                adv_stats = [(0, 1)] * self.n_advs

            for js, orig_adv in enumerate(sample_advs):
                adv = tuple(
                    (oa - mean) / (std + self.config.adv_std_eps)
                    for oa, (mean, std) in zip(orig_adv, adv_stats)
                )

                for t in trajs[sample_id]:
                    # do an in-place replacement of all graphs with buffer entries
                    G = trajs[sample_id][t][js]
                    buffer_entry = BufferEntry(
                        sample=G,
                        advantage=torch.FloatTensor(adv),
                        original_advantage=torch.FloatTensor(orig_adv),
                        # I only do features log probs no x so took that out, I know that breaks code
                        logprob_old=(G.features_logprobs).sum().cpu().item(),
                        name=f"{sample_id}___{js}___{t}",
                    )
                    trajs[sample_id][t][js] = buffer_entry

        flat_buffer = []
        for sample_id, traj in trajs.items():
            for t in traj:
                for g in traj[t]:
                    flat_buffer.append(g)

        for i in range(self.n_advs):
            all_orig_advs = torch.FloatTensor(
                [g.original_advantage[i] for g in flat_buffer]
            )
            mean, std = all_orig_advs.mean().item(), all_orig_advs.std().item()
            logger.info(
                f"Advantage {i} (ep={self._episode_counter}): {mean:.3g} +/- {std:.3g}"
            )
            self.log(name=f"Mean Train Advantage {i}", value=mean, batch_size=1)
            self.log(name=f"STD Train Advantage {i}", value=std, batch_size=1)
            self.log(
                name=f"Max Train Advantage {i}",
                value=all_orig_advs.max().item(),
                batch_size=1,
            )
            self.log(
                name=f"Min Train Advantage {i}",
                value=all_orig_advs.min().item(),
                batch_size=1,
            )

        return flat_buffer

    def get_batches(
        self, buffer: List[BufferEntry]
    ) -> Generator[Tuple[bool, List[BufferEntry]], None, None]:
        # construct batches from the buffer. yields (update_weights, batch)

        if self.config.opt.accumulate_gradients_over_time:
            assert self.config.opt.n_timesteps_per_minibatch is not None

            # reshape buffer into a dict of lists, organized by sample_id and sample number
            reshaped_buffer = defaultdict(list)
            for s in buffer:
                sample_id, js, t = s.name.split("___")
                reshaped_buffer[sample_id + "___" + js].append(s)

            keys = list(reshaped_buffer.keys())
            random.shuffle(keys)
            key_batch_size = (
                self.batch_size // self.config.opt.n_timesteps_per_minibatch
            )

            for sample_batch_idx, sample_batch_keys in enumerate(
                utils.data.make_chain(keys, size=key_batch_size)
            ):
                sample_batch = []
                for key in sample_batch_keys:
                    sample_batch.extend(reshaped_buffer[key])

                idcs = np.random.permutation(len(sample_batch))
                step_idcs = list(utils.data.make_chain(idcs, size=self.batch_size))
                n_step_batches = len(step_idcs)

                for step_batch_idx, batch in enumerate(step_idcs):
                    update_weights = (step_batch_idx == n_step_batches - 1) and (
                        (sample_batch_idx + 1)
                        % self.config.opt.gradient_accumulation_steps
                        == 0
                    )
                    yield update_weights, [sample_batch[i] for i in batch]

        else:
            idcs = np.random.permutation(len(buffer))
            batch_idcs = list(utils.data.make_chain(idcs, size=self.batch_size))

            for batch_idx, batch in enumerate(batch_idcs):
                update_weights = (
                    batch_idx + 1
                ) % self.config.opt.gradient_accumulation_steps == 0
                yield update_weights, [buffer[i] for i in batch]

    def _get_n_ppo_steps(self) -> int:
        n_ppo_steps = math.ceil(
            self.config.n_update_epochs
            * self.config.data.sampling_batch_size_per_gpu
            * self.config.data.n_samples
            / self.config.data.update_batch_size_per_gpu
            * self.model.timesteps.shape[0]
        )
        # if not self.config.opt.accumulate_gradients_over_time:
        #     n_ppo_steps *= self.model.T
        return n_ppo_steps

    def set_training_pbar(self, trainer) :
        lqdm_kwargs: Dict[str, Any] = {"desc": f"Epoch {trainer.current_epoch}: TRAIN"}
        ppo_steps = self._get_n_ppo_steps()
        loss_steps = math.ceil(
            (ppo_steps + self.config.n_diffusion_steps)
            / (1 + self.config.n_diffusion_steps)
        )
        # breakpoint()

        if self.config.n_train_steps is not None:
            lqdm_kwargs["total"] = self.config.n_train_steps * loss_steps
            lqdm_kwargs["initial"] = trainer.global_step

        elif self.config.epoch_train_steps is not None:
            lqdm_kwargs["total"] = self.config.epoch_train_steps * loss_steps

        else:
            lqdm_kwargs["total"] = (
                len(self.train_dataloader())
                * loss_steps
                # // self.config.opt.gradient_accumulation_steps
                // (trainer.num_nodes * trainer.num_devices)
            )
        self.pbar = self.lqdm(**lqdm_kwargs)


class PerSampleStatTracker:
    def __init__(self, buffer_size, n_advs: int):
        self.buffer_size = buffer_size
        self.stats = [
            defaultdict(lambda: deque(maxlen=buffer_size)) for _ in range(n_advs)
        ]

    def add_rewards(self, sample_id, rewards):
        reshaped_rewards = list(zip(*rewards))
        for i_reward, rewards in enumerate(reshaped_rewards):
            self.stats[i_reward][sample_id].extend(rewards)

    def get_stats(self, sample_id):
        stats = []
        for s in self.stats:
            mean = np.mean(s[sample_id])
            std = np.std(s[sample_id]) + 1e-6
            stats.append((mean, std))
        return stats

    def get_all_stats(self):
        return [
            {
                k: {"mean": np.mean(v), "std": np.std(v) + 1e-6, "count": len(v)}
                for k, v in s.items()
            }
            for s in self.stats
        ]
