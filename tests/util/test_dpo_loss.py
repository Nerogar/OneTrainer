import math

import torch
import torch.nn.functional as F


def _mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reproduce the per-sample MSE used in calculate_dpo_loss."""
    return (pred - target).pow(2).mean(dim=list(range(1, pred.ndim)))


def _dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the DPO loss computation from BaseModelSetup.calculate_dpo_loss."""
    chosen_ratio = policy_chosen_logp - ref_chosen_logp
    rejected_ratio = policy_rejected_logp - ref_rejected_logp
    logits = beta * (chosen_ratio - rejected_ratio)
    dpo_loss = -F.logsigmoid(logits).mean()

    if label_smoothing > 0:
        smooth_loss = -F.logsigmoid(-logits).mean()
        loss = (1 - label_smoothing) * dpo_loss + label_smoothing * smooth_loss
    else:
        loss = dpo_loss

    chosen_reward = chosen_ratio.detach().mean()
    rejected_reward = rejected_ratio.detach().mean()
    accuracy = (chosen_ratio > rejected_ratio).float().mean()

    return loss, dpo_loss, chosen_reward, rejected_reward, accuracy


class TestDPOLossMath:
    def test_beta_zero_gives_log2(self):
        """At beta=0, logits collapse to 0 and -log(sigmoid(0)) = log(2)."""
        B = 4
        policy_chosen_logp = torch.randn(B)
        policy_rejected_logp = torch.randn(B)
        ref_chosen_logp = torch.randn(B)
        ref_rejected_logp = torch.randn(B)

        loss, dpo_loss, _, _, _ = _dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=0.0,
        )
        assert abs(loss.item() - math.log(2)) < 1e-6

    def test_perfect_preference_gives_low_loss(self):
        """When policy strongly prefers chosen over rejected, loss should be low."""
        B = 4
        policy_chosen_logp = torch.tensor([0.0] * B)
        policy_rejected_logp = torch.tensor([-10.0] * B)
        ref_chosen_logp = torch.tensor([-5.0] * B)
        ref_rejected_logp = torch.tensor([-5.0] * B)

        loss, _, chosen_reward, rejected_reward, accuracy = _dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=5000.0,
        )
        assert accuracy.item() == 1.0
        assert chosen_reward.item() > rejected_reward.item()
        assert loss.item() < 0.01

    def test_inverted_preference_gives_high_loss(self):
        """When policy prefers rejected over chosen, loss should be high."""
        B = 4
        policy_chosen_logp = torch.tensor([-10.0] * B)
        policy_rejected_logp = torch.tensor([0.0] * B)
        ref_chosen_logp = torch.tensor([-5.0] * B)
        ref_rejected_logp = torch.tensor([-5.0] * B)

        loss, _, _, _, accuracy = _dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=5000.0,
        )
        assert accuracy.item() == 0.0
        assert loss.item() > 10.0

    def test_label_smoothing_reduces_extreme_loss(self):
        """Label smoothing should make loss less extreme for both directions."""
        B = 4
        policy_chosen_logp = torch.tensor([0.0] * B)
        policy_rejected_logp = torch.tensor([-10.0] * B)
        ref_chosen_logp = torch.tensor([-5.0] * B)
        ref_rejected_logp = torch.tensor([-5.0] * B)

        loss_no_smooth, _, _, _, _ = _dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=5000.0, label_smoothing=0.0,
        )
        loss_smooth, _, _, _, _ = _dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=5000.0, label_smoothing=0.1,
        )
        assert loss_smooth.item() > loss_no_smooth.item()

    def test_mse_per_sample_reduces_correctly(self):
        """MSE reduction should produce [B] shape from [B, C, H, W]."""
        B, C, H, W = 2, 4, 8, 8
        pred = torch.randn(B, C, H, W)
        target = torch.randn(B, C, H, W)
        result = _mse_per_sample(pred, target)
        assert result.shape == (B,)
        expected_0 = (pred[0] - target[0]).pow(2).mean()
        assert abs(result[0].item() - expected_0.item()) < 1e-6
