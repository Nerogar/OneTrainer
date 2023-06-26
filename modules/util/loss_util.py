from typing import Callable

import torch
from torch import Tensor


def masked_loss(
        loss_fn: Callable,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    masked_predicted = predicted * clamped_mask
    masked_target = target * clamped_mask

    losses = loss_fn(masked_predicted, masked_target, reduction="none")

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    del clamped_mask

    return losses
