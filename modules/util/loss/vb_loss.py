# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients

import torch
from torch import Tensor

import numpy as np


def normal_kl(
        mean1: Tensor,
        logvar1: Tensor,
        mean2: Tensor,
        logvar2: Tensor,
) -> Tensor:
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(
        x: Tensor,
        means: Tensor,
        log_scales: Tensor,
) -> Tensor:
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs


def __q_posterior_mean_variance(
        coefficients: DiffusionScheduleCoefficients,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
) -> (Tensor, Tensor):
    """
    Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
    """
    assert x_0.shape == x_t.shape
    posterior_mean = (
            __extract_into_tensor(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + __extract_into_tensor(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_log_variance_clipped = __extract_into_tensor(
        coefficients.posterior_log_variance_clipped, t, x_t.shape
    )
    assert (
            posterior_mean.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_0.shape[0]
    )
    return posterior_mean, posterior_log_variance_clipped


def __p_mean_variance(
        coefficients: DiffusionScheduleCoefficients,
        x_t: Tensor,
        t: Tensor,
        frozen_predicted_eps: Tensor,
        predicted_var_values: Tensor,
):
    min_log = __extract_into_tensor(coefficients.posterior_log_variance_clipped, t, x_t.shape)
    max_log = __extract_into_tensor(torch.log(coefficients.betas), t, x_t.shape)
    # The predicted_var_values is [-1, 1] for [min_var, max_var].
    frac = (predicted_var_values + 1) / 2
    predicted_log_variance = frac * max_log + (1 - frac) * min_log

    predicted_x_0 = __predict_x_0_from_eps(coefficients=coefficients, x_t=x_t, t=t, eps=frozen_predicted_eps)
    predicted_mean, _ = __q_posterior_mean_variance(coefficients=coefficients, x_0=predicted_x_0, x_t=x_t, t=t)

    return predicted_mean, predicted_log_variance


def __predict_x_0_from_eps(
        coefficients: DiffusionScheduleCoefficients,
        x_t,
        t,
        eps,
) -> Tensor:
    assert x_t.shape == eps.shape
    return (
            __extract_into_tensor(coefficients.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - __extract_into_tensor(coefficients.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )


def __vb_terms_bpd(
        coefficients: DiffusionScheduleCoefficients,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
        frozen_predicted_eps: Tensor,
        predicted_var_values: Tensor,
) -> Tensor:
    """
    Get a term for the variational lower-bound.
    The resulting units are bits (rather than nats, as one might expect).
    This allows for comparison to other papers.
    :return: a dict with the following keys:
             - 'output': a shape [N] tensor of NLLs or KLs.
             - 'pred_x_0': the x_0 predictions.
    """
    true_mean, true_log_variance_clipped = __q_posterior_mean_variance(
        x_0=x_0,
        x_t=x_t,
        t=t,
        coefficients=coefficients,
    )
    predicted_mean, predicted_log_variance = __p_mean_variance(
        coefficients=coefficients,
        x_t=x_t,
        t=t,
        frozen_predicted_eps=frozen_predicted_eps,
        predicted_var_values=predicted_var_values,
    )
    kl = normal_kl(
        true_mean, true_log_variance_clipped, predicted_mean, predicted_log_variance
    )
    kl = kl / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        x=x_0, means=predicted_mean, log_scales=0.5 * predicted_log_variance
    )
    assert decoder_nll.shape == x_0.shape
    decoder_nll = decoder_nll / np.log(2.0)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    while t.dim() < decoder_nll.dim():
        t = t.unsqueeze(-1)
    output = torch.where((t == 0), decoder_nll, kl)
    return output


def __extract_into_tensor(
        tensor: Tensor,
        timesteps: Tensor,
        broadcast_shape,
) -> Tensor:
    res = tensor[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res


def vb_losses(
        coefficients: DiffusionScheduleCoefficients,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
        predicted_eps: Tensor,
        predicted_var_values: Tensor,
) -> Tensor:
    # convert to float64 for increased precision and to prevent nan results
    # x_0 = x_0.to(dtype=torch.float64)
    # x_t = x_t.to(dtype=torch.float64)
    # predicted_eps = predicted_eps.to(dtype=torch.float64)
    # predicted_var_values = predicted_var_values.to(dtype=torch.float64)

    # Learn the variance using the variational bound, but don't let it affect our mean prediction.
    return __vb_terms_bpd(
        coefficients=coefficients,
        x_0=x_0,
        x_t=x_t,
        t=t,
        frozen_predicted_eps=predicted_eps.detach(),
        predicted_var_values=predicted_var_values,
    )
