from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.LossScaler import LossScaler
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupDiffusionLossMixin, self).__init__()
        self.align_prop_loss_fn = None
        self.coefficients = None

    def __align_prop_losses(
            self,
            batch: dict,
            data: dict,
            args: TrainArgs,
            train_device: torch.device,
    ):
        if self.align_prop_loss_fn is None:
            dtype = data['predicted'].dtype

            match args.align_prop_loss:
                case AlignPropLoss.HPS:
                    self.align_prop_loss_fn = HPSv2ScoreModel(dtype)
                case AlignPropLoss.AESTHETIC:
                    self.align_prop_loss_fn = AestheticScoreModel()

            self.align_prop_loss_fn.to(device=train_device, dtype=dtype)
            self.align_prop_loss_fn.requires_grad_(False)
            self.align_prop_loss_fn.eval()

        losses = 0

        match args.align_prop_loss:
            case AlignPropLoss.HPS:
                with torch.autocast(device_type=train_device.type, dtype=data['predicted'].dtype):
                    losses = self.align_prop_loss_fn(data['predicted'], batch['prompt'], train_device)
            case AlignPropLoss.AESTHETIC:
                losses = self.align_prop_loss_fn(data['predicted'])

        return losses * args.align_prop_weight

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            args: TrainArgs,
    ):
        losses = 0

        # MSE/L2 Loss
        if args.mse_strength != 0:
            losses += masked_losses(
                losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=args.unmasked_weight,
                normalize_masked_area_loss=args.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * args.mse_strength

        # MAE/L1 Loss
        if args.mae_strength != 0:
            losses += masked_losses(
                losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=args.unmasked_weight,
                normalize_masked_area_loss=args.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * args.mae_strength

        # VB loss
        if args.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=data['predicted'].to(dtype=torch.float32),
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=args.unmasked_weight,
                normalize_masked_area_loss=args.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * args.vb_loss_strength

        return losses

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            args: TrainArgs,
    ):
        losses = 0

        # MSE/L2 Loss
        if args.mse_strength != 0:
            losses += F.mse_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean([1, 2, 3]) * args.mse_strength

        # MAE/L1 Loss
        if args.mae_strength != 0:
            losses += F.l1_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean([1, 2, 3]) * args.mae_strength

        # VB loss
        if args.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=data['predicted'].to(dtype=torch.float32),
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean([1, 2, 3]) * args.vb_loss_strength

        if args.masked_training and args.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch['latent_mask'], args.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(dim=(1, 2, 3))
            losses /= mask_mean

        return losses

    def _diffusion_losses(
            self,
            batch: dict,
            data: dict,
            args: TrainArgs,
            train_device: torch.device,
            betas: Tensor,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        batch_size_scale = \
            1 if args.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] \
                else args.batch_size
        gradient_accumulation_steps_scale = \
            1 if args.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] \
                else args.gradient_accumulation_steps

        if self.coefficients is None:
            self.coefficients = DiffusionScheduleCoefficients.from_betas(betas)

        if data['loss_type'] == 'align_prop':
            losses = self.__align_prop_losses(batch, data, args, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if args.masked_training and not args.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, args)
            else:
                losses = self.__unmasked_losses(batch, data, args)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

        losses *= loss_weight.to(device=losses.device)
        return losses.mean()
