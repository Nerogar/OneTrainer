import math
from abc import ABCMeta

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self.__weights = None
        self._offset_noise_psi_schedule: Tensor | None = None

    def _compute_and_cache_offset_noise_psi_schedule(self, betas: Tensor) -> Tensor:
        """
        Computes the time-dependent psi_t coefficients for generalized offset noise.
        This implementation follows the paper "Generalized Diffusion Model with Adjusted Offset Noise",
        specifically Equation (34) and the logic of Algorithm 1 for the "balanced-phi_t, psi_t strategy".
        """
        if self._offset_noise_psi_schedule is not None and self._offset_noise_psi_schedule.shape[0] == betas.shape[0]:
            return self._offset_noise_psi_schedule.to(betas.device).to(torch.float64)

        betas = betas.to(torch.float64)
        T = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # From paper footnote 4: "we introduce α_0 = 1 for convenience".
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]])

        # --- Start of Algorithm 1 ---
        gammas = torch.zeros(T, device=betas.device, dtype=betas.dtype)

        # Step 1: Set gamma_1 = 1
        gammas[0] = 1.0

        # This sum is `Σ_{i=1 to t-1} γ_i/√¯αᵢ₋₁` which we build iteratively.
        cumulative_sum_term = gammas[0] / torch.sqrt(alphas_cumprod_prev[0])

        # Step 2-4: Loop for t = 2 to T (in code: t = 1 to T-1)
        for t in range(1, T):
            alpha_t = alphas[t]
            alpha_cumprod_tm1 = alphas_cumprod_prev[t]

            # Denominator from the paper's formula for C_t.
            c_t_denominator = alpha_t * (1 - alpha_cumprod_tm1)
            c_t = (1 - alpha_t) * torch.sqrt(alpha_cumprod_tm1) / c_t_denominator

            # Paper's recursive formula uses the full cumulative sum.
            gammas[t] = c_t * cumulative_sum_term

            # Update the sum for the next iteration.
            cumulative_sum_term += gammas[t] / torch.sqrt(alphas_cumprod_prev[t])

        # Step 5: Calculate normalization factor psi_T
        psi_T_denominator = torch.sqrt(1 - alphas_cumprod[-1])
        psi_T = cumulative_sum_term / psi_T_denominator

        # Step 6-8: Normalize gammas
        gammas_normalized = gammas / psi_T
        # --- End of Algorithm 1 ---

        # Finally, calculate the psi schedule for all timesteps t using Equation (22)
        terms = gammas_normalized / torch.sqrt(alphas_cumprod_prev)
        s_cumulative = torch.cumsum(terms, dim=0)
        psi_schedule = s_cumulative / torch.sqrt(1 - alphas_cumprod)

        self._offset_noise_psi_schedule = psi_schedule.to(betas.device)
        return self._offset_noise_psi_schedule


    def _create_noise(
            self,
            source_tensor: Tensor,
            config: TrainConfig,
            generator: Generator,
            timestep: Tensor | None = None,
            betas: Tensor | None = None,
    ) -> Tensor:
        noise = torch.randn(
            source_tensor.shape,
            generator=generator,
            device=config.train_device,
            dtype=source_tensor.dtype
        )

        if config.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            # Use the time-dependent generalized method if enabled.
            # This will only be true for Diffusion models (which uses betas)
            if config.generalized_offset_noise and timestep is not None and betas is not None:
                psi_schedule = self._compute_and_cache_offset_noise_psi_schedule(betas).to(timestep.device)
                psi_t = psi_schedule[timestep]
                psi_t = psi_t.view(psi_t.shape[0], *[1 for _ in range(source_tensor.ndim - 1)])
                # Scale by the time-dependent psi_t factor
                noise = noise + (psi_t * config.offset_noise_weight * offset_noise)
            else: # Otherwise, use the normal offset noise.
                noise = noise + (config.offset_noise_weight * offset_noise)

        if config.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape,
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (config.perturbation_noise_weight * perturbation_noise)

        return noise

    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            shift: float = None,
    ) -> Tensor:
        if shift is None:
            shift = config.timestep_shift

        if deterministic:
            # -1 is for zero-based indexing
            return torch.tensor(
                int(num_train_timesteps * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)
        else:
            min_timestep = int(num_train_timesteps * config.min_noising_strength)
            max_timestep = int(num_train_timesteps * config.max_noising_strength)
            num_timestep = max_timestep - min_timestep

            if config.timestep_distribution in [
                TimestepDistribution.UNIFORM,
                TimestepDistribution.LOGIT_NORMAL,
                TimestepDistribution.HEAVY_TAIL
            ]:
                # continuous implementations
                if config.timestep_distribution == TimestepDistribution.UNIFORM:
                    timestep = min_timestep + (max_timestep - min_timestep) \
                               * torch.rand(batch_size, generator=generator, device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
                    bias = config.noising_bias
                    scale = config.noising_weight + 1.0

                    normal = torch.normal(bias, scale, size=(batch_size,), generator=generator, device=generator.device)
                    logit_normal = normal.sigmoid()
                    timestep = logit_normal * num_timestep + min_timestep
                elif config.timestep_distribution == TimestepDistribution.HEAVY_TAIL:
                    scale = config.noising_weight

                    u = torch.rand(
                        size=(batch_size,),
                        generator=generator,
                        device=generator.device,
                    )
                    u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
                    timestep = u * num_timestep + min_timestep

                timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
            else:
                # Shifting a discrete distribution is done in two steps:
                # 1. Apply the inverse shift to the linspace.
                #    This moves the sample points of the function to their shifted place.
                # 2. Multiply the result with the derivative of the inverse shift function.
                #    The derivative is an approximation of the distance between sample points.
                #    Or in other words, the size of a shifted bucket in the original function.
                linspace = torch.linspace(0, 1, num_timestep)
                linspace = linspace / (shift - shift * linspace + linspace)

                linspace_derivative = torch.linspace(0, 1, num_timestep)
                linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

                # continuous implementations
                if config.timestep_distribution == TimestepDistribution.COS_MAP:
                    if self.__weights is None:
                        weights = 2.0 / (math.pi - 2.0 * math.pi * linspace + 2.0 * math.pi * linspace ** 2.0)
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.SIGMOID:
                    if self.__weights is None:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = linspace / (shift - shift * linspace + linspace)
                        weights = 1 / (1 + torch.exp(-weight * (weights - bias)))  # Sigmoid
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.INVERTED_PARABOLA:
                    if self.__weights is None:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = torch.clamp(-weight * ((linspace - bias) ** 2) + 2, min=0.0)
                        weights *= linspace_derivative
                        self.__weights = weights.to(device=generator.device)
                samples = torch.multinomial(self.__weights, num_samples=batch_size, replacement=True, generator=generator) + min_timestep
                timestep = samples.to(dtype=torch.long, device=generator.device)

            return timestep.int()

    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        if deterministic:
            return torch.full(
                size=(batch_size,),
                fill_value=0.5,
                device=generator.device,
            )
        else:
            discrete_timesteps = 10000  # Discretize to 10000 timesteps
            discrete = self._get_timestep_discrete(
                num_train_timesteps=discrete_timesteps,
                deterministic=False,
                generator=generator,
                batch_size=batch_size,
                config=config,
            ) + 1

            continuous = (discrete.float() / discrete_timesteps)
            return continuous
