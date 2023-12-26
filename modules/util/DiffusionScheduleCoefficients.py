import torch
from torch import Tensor


class DiffusionScheduleCoefficients:
    def __init__(
            self,
            num_timesteps,
            betas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
    ):
        self.num_timesteps = num_timesteps
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.log_one_minus_alphas_cumprod = log_one_minus_alphas_cumprod
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = posterior_log_variance_clipped
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2

    @staticmethod
    def from_betas(betas: Tensor):
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.tensor([1], dtype=alphas_cumprod.dtype, device=betas.device), alphas_cumprod[:-1]))
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]).clamp(min=1e-20)
        )

        return DiffusionScheduleCoefficients(
            num_timesteps=betas.shape[0],
            betas=betas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1 - alphas_cumprod),
            log_one_minus_alphas_cumprod=torch.log(1 - alphas_cumprod),
            sqrt_recip_alphas_cumprod=torch.rsqrt(alphas_cumprod),
            sqrt_recipm1_alphas_cumprod=torch.sqrt(1 / alphas_cumprod - 1),
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            posterior_mean_coef1=(betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
            posterior_mean_coef2=((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )
