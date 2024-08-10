import torch
from diffusers import DDIMScheduler


def rescale_noise_scheduler_to_zero_terminal_snr(noise_scheduler: DDIMScheduler):
    """
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed (https://arxiv.org/abs/2305.08891)

    Rescales the

    Args:
        noise_scheduler: The noise scheduler to transform

    Returns:

    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5

    # Store old values.
    alphas_cumprod_sqrt_0 = sqrt_alphas_cumprod[0].clone()
    alphas_cumprod_sqrt_T = sqrt_alphas_cumprod[-1].clone()

    # Shift so last timestep is zero.
    sqrt_alphas_cumprod -= alphas_cumprod_sqrt_T

    # Scale so first timestep is back to old value.
    sqrt_alphas_cumprod *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

    # Convert alphas_cumprod_sqrt to betas
    alphas_cumprod = sqrt_alphas_cumprod ** 2
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod

    return betas
