import torch
from torch import Tensor


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    return losses


def masked_losses_with_prior(
        losses: Tensor,
        prior_losses: Tensor | None,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        masked_prior_preservation_weight: float,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    if masked_prior_preservation_weight == 0 or prior_losses is None:
        return losses

    clamped_mask = (1 - clamped_mask)
    prior_losses *= clamped_mask * masked_prior_preservation_weight

    if normalize_masked_area_loss:
        prior_losses = prior_losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    return losses + prior_losses
