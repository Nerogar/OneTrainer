import torch
import torch.nn.functional as F
from torch import Tensor

from modules.util.enum.DistillationLossType import DistillationLossType


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


def distillation_loss(
        student_prediction: Tensor,
        parent_prediction: Tensor,
        loss_type: DistillationLossType,
        temperature: float = 1.0,
        mask: Tensor | None = None,
        reduction: str = 'mean',
) -> Tensor:
    """
    Calculate distillation loss between student and parent predictions.
    
    Args:
        student_prediction: Student model output
        parent_prediction: Parent model output (should be detached)
        loss_type: Type of loss (MSE, MAE, Huber, KL)
        temperature: Temperature scaling for KL divergence
        mask: Optional mask for masked training
        reduction: 'mean' or 'none'
        
    Returns:
        Distillation loss tensor
    """
    # Ensure parent prediction is detached (no gradients through parent)
    parent_prediction = parent_prediction.detach()
    
    if loss_type == DistillationLossType.MSE:
        loss = F.mse_loss(student_prediction, parent_prediction, reduction='none')
    elif loss_type == DistillationLossType.MAE:
        loss = F.l1_loss(student_prediction, parent_prediction, reduction='none')
    elif loss_type == DistillationLossType.HUBER:
        loss = F.huber_loss(student_prediction, parent_prediction, reduction='none', delta=1.0)
    elif loss_type == DistillationLossType.KL_DIVERGENCE:
        # For KL divergence on continuous predictions (e.g., latents)
        # We use a log-normal approximation:
        # Treat predictions as means of distributions, compute KL between Gaussians
        # KL(P||Q) = 0.5 * (||mu_p - mu_q||^2 / sigma^2)
        # For simplicity, assume unit variance (sigma=1)
        # Temperature scaling: divide by temperature before computing
        student_scaled = student_prediction / temperature
        parent_scaled = parent_prediction / temperature
        
        # MSE-based KL approximation with temperature scaling
        loss = F.mse_loss(student_scaled, parent_scaled, reduction='none') * (temperature ** 2)
    else:
        raise ValueError(f"Unknown distillation loss type: {loss_type}")
    
    # Apply mask if provided
    if mask is not None:
        # Use same masking logic as other losses
        clamped_mask = torch.clamp(mask, 0.1, 1.0)  # Use default unmasked_weight
        loss = loss * clamped_mask
    
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss
