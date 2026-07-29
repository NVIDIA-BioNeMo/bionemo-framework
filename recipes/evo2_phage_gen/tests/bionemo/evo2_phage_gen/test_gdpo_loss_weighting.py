import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn


def _loss(*, token_level_loss: bool) -> float:
    token_mask = torch.tensor(
        [[0, 1, 1, 1, 1], [0, 1, 0, 0, 0]], dtype=torch.float32
    )
    advantages = torch.tensor(
        [[0, 1, 1, 1, 1], [0, -1, -1, -1, -1]], dtype=torch.float32
    )
    zeros = torch.zeros_like(advantages)
    data = {
        "advantages": advantages,
        "prev_logprobs": zeros,
        "generation_logprobs": zeros,
        "reference_policy_logprobs": zeros,
        "token_mask": token_mask,
        "sample_mask": torch.ones(2),
    }
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=token_level_loss,
            reference_policy_kl_penalty=0,
            force_on_policy_ratio=True,
        )
    )
    loss, _ = loss_fn(
        torch.zeros((2, 4)),
        data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(5.0),
    )
    return loss.item()


def test_sequence_loss_does_not_discount_short_negative_response():
    assert _loss(token_level_loss=True) == pytest.approx(-0.6)
    assert _loss(token_level_loss=False) == pytest.approx(0.0)
