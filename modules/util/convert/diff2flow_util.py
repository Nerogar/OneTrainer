from functools import partial
from types import MethodType
from typing import Any

import torch

import numpy as np


def enable_diff2flow(model):
    """
    Modifies the model in-place to enable training and inference
    with the Diff2Flow methodology.
    The logic based on the paper:
    "Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment"
    Modified from:
    https://github.com/CompVis/diff2flow/blob/33239aa0c02c554ee0b3fff5c5f0167a8dabdf6a/diff2flow/flow_obj.py
    """
    if hasattr(model, 'get_diff2flow_velocity'):
        return  # Already enabled

    device = model.unet.device

    # Pre-compute and register buffers needed for trajectory conversion
    betas = model.noise_scheduler.betas.cpu().numpy()
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_full = np.append(1., alphas_cumprod)

    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

    # Store base constants
    model.df_sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
    model.df_sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))

    # Store required root and complementary root constants (with full array, including t=0)
    model.df_sqrt_alphas_cumprod_full = to_torch(np.sqrt(alphas_cumprod_full))
    model.df_sqrt_one_minus_alphas_cumprod_full = to_torch(np.sqrt(1. - alphas_cumprod_full))

    # Store reciprocal root constants
    model.df_sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
    model.df_sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1))

    # Store 'rectified' constants used for Flow-to-Diffusion timestep conversion
    model.df_rectified_alphas_cumprod_full = model.df_sqrt_alphas_cumprod_full / (model.df_sqrt_alphas_cumprod_full + model.df_sqrt_one_minus_alphas_cumprod_full)

    # Bind methods to the model instance for easy access
    model._df_extract_into_tensor = MethodType(_df_extract_into_tensor, model)
    model._df_convert_fm_t_to_dm_t = MethodType(_df_convert_fm_t_to_dm_t, model)
    model._df_convert_fm_xt_to_dm_xt = MethodType(_df_convert_fm_xt_to_dm_xt, model)
    model._df_predict_start_from_z_and_v = MethodType(_df_predict_start_from_z_and_v, model)
    model._df_predict_eps_from_z_and_v = MethodType(_df_predict_eps_from_z_and_v, model)
    model._df_predict_start_from_eps = MethodType(_df_predict_start_from_eps, model)
    model._df_get_vector_field_from_v = MethodType(_df_get_vector_field_from_v, model)
    model._df_get_vector_field_from_eps = MethodType(_df_get_vector_field_from_eps, model)
    model.get_diff2flow_velocity = MethodType(get_diff2flow_velocity, model)

def _df_extract_into_tensor(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """
    a: The tensor to extract from (e.g., self.df_sqrt_alphas_cumprod)
    t: The continuous timestep tensor.
    x_shape: The shape of the target tensor (x_t) for reshaping.
    """
    b, *_ = t.shape
    # t can be float here, linearly interpolate between left and right index
    t = t.to(a.device)
    t = t.clamp(0, a.shape[-1] - 1)

    # Linear interpolation logic
    left_idx = t.long()
    right_idx = (left_idx + 1).clamp(max=a.shape[-1] - 1)

    left_val = a.gather(-1, left_idx)
    right_val = a.gather(-1, right_idx)
    t_ = t - left_idx.float()

    out = left_val * (1 - t_) + right_val * t_

    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _df_convert_fm_t_to_dm_t(self, t: torch.Tensor) -> torch.Tensor:
    """
    Converts 'flow' (fm) timestep to 'diffusion' (dm) timestep.
    """
    rectified_alphas_cumprod_full = self.df_rectified_alphas_cumprod_full.clone().to(t.device)
    rectified_alphas_cumprod_full = torch.flip(rectified_alphas_cumprod_full, [0])
    right_index = torch.searchsorted(rectified_alphas_cumprod_full, t, right=True)
    left_index = right_index - 1
    right_value = rectified_alphas_cumprod_full[right_index]
    left_value = rectified_alphas_cumprod_full[left_index]
    dm_t = left_index.float() + (t - left_value) / (right_value - left_value)
    dm_t = self.noise_scheduler.config.num_train_timesteps - dm_t
    return dm_t


def _df_convert_fm_xt_to_dm_xt(self, fm_xt: torch.Tensor, fm_t: torch.Tensor) -> torch.Tensor:
    """
    Converts 'flow' (fm) latent variable to 'diffusion' (dm) latent variable
    by applying a time-dependent scaling.
    """
    scale = self.df_sqrt_alphas_cumprod_full + self.df_sqrt_one_minus_alphas_cumprod_full

    dm_t = self._df_convert_fm_t_to_dm_t(fm_t)

    # Interpolate the scale factor based on the continuous diffusion timestep
    dm_t_left_index = torch.floor(dm_t)
    dm_t_right_index = torch.ceil(dm_t)

    # Clamp indices just in case the conversion pushes them out of bounds
    max_idx = scale.shape[-1] - 1
    dm_t_left_index = dm_t_left_index.clamp(0, max_idx).long()
    dm_t_right_index = dm_t_right_index.clamp(0, max_idx).long()

    dm_t_left_value = scale[dm_t_left_index]
    dm_t_right_value = scale[dm_t_right_index]

    # Linear interpolation for scale
    scale_t = dm_t_left_value + (dm_t - dm_t_left_index.float()) * (dm_t_right_value - dm_t_left_value)
    scale_t = scale_t.view(-1, 1, 1, 1)
    dm_xt = fm_xt * scale_t
    return dm_xt


def _df_predict_start_from_z_and_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Predicts the initial (clean) image `x_0` (or `z_0` in latent space) from
    the noisy input `x_t` and the predicted velocity `v`.
    """
    return (
        self._df_extract_into_tensor(self.df_sqrt_alphas_cumprod, t, x_t.shape) * x_t -
        self._df_extract_into_tensor(self.df_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
    )


def _df_predict_eps_from_z_and_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Predicts the noise component `epsilon` from the noisy input `x_t` and the
    predicted velocity `v`.
    """
    return (
        self._df_extract_into_tensor(self.df_sqrt_alphas_cumprod, t, x_t.shape) * v +
        self._df_extract_into_tensor(self.df_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
    )


def _df_predict_start_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Predicts the initial (clean) image `x_0` from the noisy input `x_t` and the
    predicted noise `epsilon`. This is the standard diffusion model conversion.
    """
    return (
        self._df_extract_into_tensor(self.df_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._df_extract_into_tensor(self.df_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )


def _df_get_vector_field_from_v(self, v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculates the vector field from the predicted velocity.
    Returns: Noise - Data
    """
    z_pred = self._df_predict_start_from_z_and_v(x_t, t, v)
    eps_pred = self._df_predict_eps_from_z_and_v(x_t, t, v)
    vector_field = eps_pred - z_pred
    return vector_field


def _df_get_vector_field_from_eps(self, noise: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Calculates the vector field from the predicted noise.
    Returns: Noise - Data
    """
    z_pred = self._df_predict_start_from_eps(x_t, t, noise)
    eps_pred = noise
    vector_field = eps_pred - z_pred
    return vector_field


@torch.no_grad()
def get_diff2flow_velocity(self, fm_x: torch.Tensor, fm_t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """
    The main inference/sampling function.
    Predicts the vector field (Diff2Flow velocity) using the underlying UNet model.
    """
    # Convert Flow-Matching (FM) inputs to Diffusion Model (DM) inputs
    dm_t_continuous = self._df_convert_fm_t_to_dm_t(fm_t)
    dm_x = self._df_convert_fm_xt_to_dm_xt(fm_x, fm_t)

    # Run the UNet with DM inputs
    model_pred = self.unet(
        dm_x,
        dm_t_continuous,
        **kwargs,
        return_dict=False
    )[0]

    # Handle NaN prediction (as in original code)
    if torch.isnan(model_pred).any():
        model_pred[torch.isnan(model_pred)] = 0

    # Convert UNet prediction (v-pred or epsilon) to the vector field
    if self.noise_scheduler.config.prediction_type == 'v_prediction':
        vector_field = self._df_get_vector_field_from_v(model_pred, dm_x, dm_t_continuous)
    elif self.noise_scheduler.config.prediction_type == 'epsilon':
        vector_field = self._df_get_vector_field_from_eps(model_pred, dm_x, dm_t_continuous)

    return vector_field
