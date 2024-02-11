from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.LossScaler import LossScaler
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupDiffusionLossMixin, self).__init__()
        self.__align_prop_loss_fn = None
        self.__coefficients = None

    def __align_prop_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
    ):
        if self.__align_prop_loss_fn is None:
            dtype = data['predicted'].dtype

            match config.align_prop_loss:
                case AlignPropLoss.HPS:
                    self.__align_prop_loss_fn = HPSv2ScoreModel(dtype)
                case AlignPropLoss.AESTHETIC:
                    self.__align_prop_loss_fn = AestheticScoreModel()

            self.__align_prop_loss_fn.to(device=train_device, dtype=dtype)
            self.__align_prop_loss_fn.requires_grad_(False)
            self.__align_prop_loss_fn.eval()

        losses = 0

        match config.align_prop_loss:
            case AlignPropLoss.HPS:
                with torch.autocast(device_type=train_device.type, dtype=data['predicted'].dtype):
                    losses = self.__align_prop_loss_fn(data['predicted'], batch['prompt'], train_device)
            case AlignPropLoss.AESTHETIC:
                losses = self.__align_prop_loss_fn(data['predicted'])

        return losses * config.align_prop_weight

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ):
        losses = 0

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += masked_losses(
                losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += masked_losses(
                losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.mae_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=data['predicted'].to(dtype=torch.float32),
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.vb_loss_strength

        return losses

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ):
        losses = 0

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += F.mse_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean([1, 2, 3]) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += F.l1_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean([1, 2, 3]) * config.mae_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.__coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=data['predicted'].to(dtype=torch.float32),
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean([1, 2, 3]) * config.vb_loss_strength

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch['latent_mask'], config.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(dim=(1, 2, 3))
            losses /= mask_mean

        return losses

    def __min_snr_weight(
            self,
            timesteps: Tensor,
            gamma: int,
            device: torch.device
    ):
        all_snr = (self.__coefficients.sqrt_alphas_cumprod /
                   self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
        all_snr.to(device)
        snr = torch.stack([all_snr[t] for t in timesteps])
        gamma_over_snr = torch.div(torch.ones_like(snr)*gamma, snr)
        snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float().to(device)
        return snr_weight

    def _diffusion_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            betas: Tensor | None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        batch_size_scale = \
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] \
                else config.batch_size
        gradient_accumulation_steps_scale = \
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] \
                else config.gradient_accumulation_steps

        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas)

        if data['loss_type'] == 'align_prop':
            losses = self.__align_prop_losses(batch, data, config, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

        losses *= loss_weight.to(device=losses.device)

        # Apply minimum SNR weighting.
        if config.min_snr_gamma:
            snr_weight = self.__min_snr_weight(data['timestep'], config.min_snr_gamma, losses.device)
            losses *= snr_weight

        return losses
