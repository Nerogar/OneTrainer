from functools import partial
from types import MethodType

import torch

import numpy as np


def enable_diff2flow(model):
    """
    Modifies the model in-place to enable training and inference
    with the Diff2Flow methodology.
    The logic based on the paper:
    "Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment"
    And copied from:
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

    model.df_alphas_cumprod = to_torch(alphas_cumprod)
    model.df_alphas_cumprod_full = to_torch(alphas_cumprod_full)

    sqrt_alphas_cumprod_full = np.sqrt(alphas_cumprod_full)
    sqrt_one_minus_alphas_cumprod_full = np.sqrt(1. - alphas_cumprod_full)

    model.df_sqrt_alphas_cumprod_full = to_torch(sqrt_alphas_cumprod_full)
    model.df_sqrt_one_minus_alphas_cumprod_full = to_torch(sqrt_one_minus_alphas_cumprod_full)

    rectified_alphas_cumprod_full = sqrt_alphas_cumprod_full / (
            sqrt_alphas_cumprod_full + sqrt_one_minus_alphas_cumprod_full)
    model.df_rectified_alphas_cumprod_full = to_torch(rectified_alphas_cumprod_full)

    # Temporarily allow division by zero to handle the last timestep in zero-SNR schedules
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

    # Replace any resulting 'inf' values with 0.
    sqrt_recip_alphas_cumprod[np.isinf(sqrt_recip_alphas_cumprod)] = 0
    sqrt_recipm1_alphas_cumprod[np.isinf(sqrt_recipm1_alphas_cumprod)] = 0
    model.df_sqrt_recip_alphas_cumprod = to_torch(sqrt_recip_alphas_cumprod)
    model.df_sqrt_recipm1_alphas_cumprod = to_torch(sqrt_recipm1_alphas_cumprod)

    # Bind methods to the model instance for easy access
    model._df_convert_fm_t_to_dm_t = MethodType(_df_convert_fm_t_to_dm_t, model)
    model._df_convert_fm_xt_to_dm_xt = MethodType(_df_convert_fm_xt_to_dm_xt, model)
    model._df_predict_start_from_z_and_v = MethodType(_df_predict_start_from_z_and_v, model)
    model._df_predict_eps_from_z_and_v = MethodType(_df_predict_eps_from_z_and_v, model)
    model._df_predict_start_from_eps = MethodType(_df_predict_start_from_eps, model)
    model._df_get_vector_field_from_v = MethodType(_df_get_vector_field_from_v, model)
    model._df_get_vector_field_from_eps = MethodType(_df_get_vector_field_from_eps, model)
    model.get_diff2flow_velocity = MethodType(get_diff2flow_velocity, model)

def extract_into_tensor(a, t, x_shape):
    """
    Extracts values from a 1D tensor for a batch of indices.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def _df_convert_fm_t_to_dm_t(self, t):
    """ Converts continuous Flow Matching time [0,1] to discrete Diffusion Model time [0, T-1]. """
    rectified_alphas_cumprod_full = self.df_rectified_alphas_cumprod_full.clone().to(t.device)
    rectified_alphas_cumprod_full = torch.flip(rectified_alphas_cumprod_full, [0])
    right_index = torch.searchsorted(rectified_alphas_cumprod_full, t, right=True)
    left_index = right_index - 1
    right_value = rectified_alphas_cumprod_full[right_index]
    left_value = rectified_alphas_cumprod_full[left_index]
    dm_t = left_index + (t - left_value) / (right_value - left_value)
    dm_t = self.noise_scheduler.config.num_train_timesteps - dm_t
    return dm_t

def _df_convert_fm_xt_to_dm_xt(self, fm_xt, fm_t):
    """ Converts an interpolant on the FM trajectory to its equivalent on the DM trajectory. """
    scale = self.df_sqrt_alphas_cumprod_full + self.df_sqrt_one_minus_alphas_cumprod_full
    dm_t = self._df_convert_fm_t_to_dm_t(fm_t)

    dm_t_left_index = torch.floor(dm_t).long()
    dm_t_right_index = torch.ceil(dm_t).long()
    dm_t_left_value = scale[dm_t_left_index].to(fm_xt.device)
    dm_t_right_value = scale[dm_t_right_index].to(fm_xt.device)

    scale_t = dm_t_left_value + (dm_t - dm_t_left_index) * (dm_t_right_value - dm_t_left_value)
    scale_t = scale_t.view(-1, 1, 1, 1)
    dm_xt = fm_xt * scale_t
    return dm_xt

def _df_predict_start_from_z_and_v(self, x_t, t, v):
    return (
        extract_into_tensor(self.df_sqrt_alphas_cumprod_full[1:], t, x_t.shape) * x_t -
        extract_into_tensor(self.df_sqrt_one_minus_alphas_cumprod_full[1:], t, x_t.shape) * v
    )

def _df_predict_eps_from_z_and_v(self, x_t, t, v):
    return (
        extract_into_tensor(self.df_sqrt_alphas_cumprod_full[1:], t, x_t.shape) * v +
        extract_into_tensor(self.df_sqrt_one_minus_alphas_cumprod_full[1:], t, x_t.shape) * x_t
    )

def _df_predict_start_from_eps(self, x_t, t, noise):
    return (
        extract_into_tensor(self.df_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract_into_tensor(self.df_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

def _df_get_vector_field_from_v(self, v, x_t, t):
    z_pred = self._df_predict_start_from_z_and_v(x_t, t, v)
    eps_pred = self._df_predict_eps_from_z_and_v(x_t, t, v)
    vector_field = z_pred - eps_pred
    return vector_field

def _df_get_vector_field_from_eps(self, noise, x_t, t):
    z_pred = self._df_predict_start_from_eps(x_t, t, noise)
    eps_pred = noise
    vector_field = z_pred - eps_pred
    return vector_field

@torch.no_grad()
def get_diff2flow_velocity(self, fm_x, fm_t, **kwargs):
    dm_t_continuous = self._df_convert_fm_t_to_dm_t(fm_t)

    # Clamp the discrete timestep to the valid range [0, num_train_timesteps - 1]
    dm_t_discrete = dm_t_continuous.round().long().clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)

    dm_x = self._df_convert_fm_xt_to_dm_xt(fm_x, fm_t)

    model_pred = self.unet(
        dm_x,
        dm_t_discrete,
        **kwargs,
        return_dict=False
    )[0]

    if torch.isnan(model_pred).any():
        model_pred[torch.isnan(model_pred)] = 0

    if self.noise_scheduler.config.prediction_type == 'v_prediction':
        vector_field = self._df_get_vector_field_from_v(model_pred, dm_x, dm_t_discrete)
    elif self.noise_scheduler.config.prediction_type == 'epsilon':
        vector_field = self._df_get_vector_field_from_eps(model_pred, dm_x, dm_t_discrete)

    return vector_field
