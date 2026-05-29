import torch
from typing import *
import yaml
from torch.optim.lr_scheduler import LambdaLR

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
LongType = Union[int, LongTensor]

TRAIN = 0
VALIDATION = 1
INFERENCE = 2
SAMPLING = 3

N_AA = 21

def t2n(x):
    """Convert torch tensor(s) to numpy array(s)

    Args:
        x (Any): Must be a tensor, a list of tensors, or a tensor-valued dict.
            Compositions of the above are allowed, for example a list of tensor-valued
            dicts, or a dict mapping strings to list of tensors. Anything that doesn't
            conform to the above (e.g. None or np.array) will be returned unchanged.

    Returns:
        Something numerical.
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if len(x.shape) == 0:
            return x.item()
        else:
            return x.numpy()
    elif isinstance(x, list):
        return [t2n(xx) for xx in x]
    elif isinstance(x, dict):
        return {k: t2n(v) for k, v in x.items()}
    else:
        return x

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(config)

def get_polynomial_learning_rate(
    optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    train_dataloader,
    step_scale_factor: float,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):

    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step: int):
        if num_training_steps is None:
            try:
                num_training_steps_final = len(train_dataloader) * step_scale_factor
            except TypeError:
                num_training_steps_final = step_scale_factor
            num_training_steps_final = int(num_training_steps_final)
        else:
            num_training_steps_final = num_training_steps

        # print("NUM TRAINING STEPS", num_training_steps_final)

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps_final:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps_final - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)
