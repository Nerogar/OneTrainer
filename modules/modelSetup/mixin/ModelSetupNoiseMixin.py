import math
from abc import ABCMeta
from collections import defaultdict

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self._timestep_probability_cache = TimestepProbabilityCache()
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
            batch_config: dict[str, Tensor],
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

        offset_noise_weight = batch_config["offset_noise_weight"]
        if torch.any(offset_noise_weight != 0):
            offset_noise = torch.randn(
                (*source_tensor.shape[:2], *[1]*(source_tensor.ndim - 2)),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )

            offset_noise_weight = offset_noise_weight.view(-1, *[1]*(offset_noise.ndim - 1)).to(config.train_device)

            # Use the time-dependent generalized method if enabled.
            # This will only be true for Diffusion models (which uses betas)
            if config.generalized_offset_noise and timestep is not None and betas is not None:
                psi_schedule = self._compute_and_cache_offset_noise_psi_schedule(betas).to(timestep.device)
                psi_t = psi_schedule[timestep]
                psi_t = psi_t.view(psi_t.shape[0], *[1]*(source_tensor.ndim - 1))
                # Scale by the time-dependent psi_t factor
                noise += psi_t * offset_noise_weight * offset_noise
            else: # Otherwise, use the normal offset noise.
                noise += offset_noise_weight * offset_noise

        perturbation_noise_weight = batch_config["perturbation_noise_weight"]
        if torch.any(perturbation_noise_weight != 0):
            perturbation_noise = torch.randn(
                source_tensor.shape,
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )

            perturbation_noise_weight = perturbation_noise_weight \
                .view(-1, *[1]*(perturbation_noise.ndim - 1)).to(config.train_device)

            noise += perturbation_noise_weight * perturbation_noise

        return noise


    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            batch_config: dict[str, Tensor],
            shift: float = None,
    ) -> Tensor:
        shift: Tensor
        if shift is not None and config.dynamic_timestep_shifting:
            shift = torch.tensor(shift).expand(batch_config["timestep_shift"].shape)
        else:
            shift = batch_config["timestep_shift"]

        if deterministic:
            # -1 is for zero-based indexing
            return torch.tensor(
                int(num_train_timesteps * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)

        self._timestep_probability_cache.maintenance_step(config.timestep_distribution)

        # check number of different settings
        cache_keys = TimestepProbabilityCache.get_keys(batch_config, shift)
        if len(set(cache_keys)) == 1:
            cache_keys = cache_keys[:1]

        # gather cumulative probabilities for each sample from cache, or regenerate
        batch_cumprobs = []
        for i, cache_key in enumerate(cache_keys):
            cache_entry = self._timestep_probability_cache[cache_key]
            if cache_entry.cumprobs is None:
                cache_entry.cumprobs = self.__prepare_timestep_cumprobs(
                    num_train_timesteps,
                    config.timestep_distribution,
                    batch_config,
                    shift,
                    i,
                ).to(generator.device)

            batch_cumprobs.append(cache_entry.cumprobs)

        # sample timesteps with binary search over normalized cumulative probabilities
        u = torch.rand(batch_size, generator=generator, device=generator.device)
        if len(batch_cumprobs) == 1:
            cumprobs = batch_cumprobs[0]
        else:
            cumprobs = torch.stack(batch_cumprobs)
            u = u.unsqueeze(1)

        timestep = torch.searchsorted(cumprobs, u, out_int32=True, right=True).view(-1)
        return timestep

    def __prepare_timestep_cumprobs(
            self,
            num_train_timesteps: int,
            timestep_distribution: TimestepDistribution,
            batch_config: dict[str, Tensor],
            batch_shift: Tensor,
            index: int,
    ) -> Tensor:
        min_strength = batch_config["min_noising_strength"][index]
        max_strength = batch_config["max_noising_strength"][index]
        bias  = float(batch_config["noising_bias"][index])
        scale = float(batch_config["noising_weight"][index])
        shift = max(float(batch_shift[index]), 1e-10)

        min_timestep = int(num_train_timesteps * min_strength.clamp(0, 1))
        max_timestep = int(num_train_timesteps * max_strength.clamp(0, 1))

        if min_timestep > max_timestep:
            min_timestep = max_timestep
            print(f"Warning: min noising strength ({min_strength:.3f}) > max noising strength ({max_strength:.3f})")

        if min_timestep == max_timestep:
            if min_timestep > 0:
                min_timestep -= 1
            else:
                max_timestep += 1

        num_timestep = max_timestep - min_timestep

        # Shifting a discrete distribution is done in two steps:
        # 1. Apply the inverse shift to the linspace.
        #    This moves the sample points of the function to their shifted place.
        # 2. Multiply the result with the derivative of the inverse shift function.
        #    The derivative is an approximation of the distance between sample points.
        #    Or in other words, the size of a shifted bucket in the original function.

        half_bucket = 0.5 / num_timestep  # Sample at the center of buckets

        # Use double precision to avoid zeroing-out small trailing weights in cumsum below
        # due to insufficient accuracy when small values are added to a large running sum.
        linspace = torch.linspace(half_bucket, 1-half_bucket, num_timestep, dtype=torch.float64)
        linspace = linspace / (shift - shift * linspace + linspace)

        linspace_derivative = torch.linspace(half_bucket, 1-half_bucket, num_timestep, dtype=torch.float64)
        linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

        # Plot: https://www.desmos.com/calculator/q88f7b9wda
        if timestep_distribution == TimestepDistribution.UNIFORM:
            weights = torch.full_like(linspace, 1)
        elif timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
            scale = max(scale + 1.0, 0.001)
            weights = (1.0 / (scale * math.sqrt(2.0 * math.pi)))  \
                    * (1.0 / (linspace * (1.0 - linspace)))       \
                    * torch.exp( -((torch.logit(linspace) - bias) ** 2.0) / (2.0 * scale ** 2.0) )
            weights.nan_to_num_(0)
        elif timestep_distribution == TimestepDistribution.COS_MAP:
            weights = 2.0 / (math.pi - (2.0 * math.pi * linspace) + (2.0 * math.pi * linspace ** 2.0))
        elif timestep_distribution == TimestepDistribution.SIGMOID:
            bias += 0.5
            weights = 1 / (1 + torch.exp(-scale * (linspace - bias)))
        elif timestep_distribution == TimestepDistribution.INVERTED_PARABOLA:
            bias += 0.5
            weights = torch.clamp(-scale * ((linspace - bias) ** 2) + 2, min=0.0)
        elif timestep_distribution == TimestepDistribution.HEAVY_TAIL:
            scale = min(scale, 1.735)  # The approximation breaks when the quantile function becomes non-monotonic

            def quantile(x):
                return 1 - x - scale * (torch.cos(math.pi / 2 * x) ** 2 - 1 + x)
            def derivative(x):
                return -(1 + scale) + scale * (math.pi / 2) * torch.sin(math.pi * x)

            # Use Newton's method to approximate the quantile function and evaluate the probability density
            x = 1.0 - linspace  # Initial guess matches quantile() for scale=0
            for _ in range(20):
                delta = (quantile(x) - linspace) / derivative(x)
                x -= delta
                if delta.abs().max() < 1e-10:
                    break

            weights = 1 / derivative(x).abs()
        else:
            raise ValueError(f"Unknown timestep distribution: {timestep_distribution}")

        weights *= linspace_derivative

        if num_timestep != num_train_timesteps:
            weights_padded = torch.zeros(num_train_timesteps, dtype=torch.float64)
            weights_padded[min_timestep:max_timestep] = weights
            weights = weights_padded

        cumprobs = weights.cumsum_(0)
        cumprobs /= cumprobs[-1].item() # normalize
        cumprobs[-1] = 1 # avoid sampling out of range when all weights are 0
        return cumprobs.float()


    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            batch_config: dict,
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
                batch_config=batch_config
            ) + 1

            continuous = (discrete.float() / discrete_timesteps)
            return continuous


class TimestepProbabilityCache:
    """
    Caches the cumulative probabilities of timesteps.

    The cache key consists of all relevant settings for calculating the timestep probabilities.
    A user can change these settings through the UI during training, causing new cache entries to be generated.
    Therefore, old cache entries are removed when unused for 100 training steps. This check happens every 20 steps.

    When the timestep distribution function itself is changed, the cache is cleared.
    """

    class Entry:
        def __init__(self):
            self.cumprobs: Tensor | None = None
            self.last_use_step = 0

    def __init__(self):
        self.cache = defaultdict(self.Entry)
        self.timestep_distribution: TimestepDistribution | None = None
        self.step = 0

    def maintenance_step(self, timestep_distribution: TimestepDistribution):
        if timestep_distribution != self.timestep_distribution:
            self.timestep_distribution = timestep_distribution
            self.cache.clear()
            self.step = 1
        else:
            self.step += 1
            if not (self.step % 20):
                self.evict_stale_entries()

    def evict_stale_entries(self):
        step_limit = self.step - 100
        remove_keys = [
            k for k, entry in self.cache.items()
            if entry.last_use_step < step_limit
        ]
        for k in remove_keys:
            self.cache.pop(k, None)

    def __getitem__(self, key: tuple) -> 'TimestepProbabilityCache.Entry':
        entry = self.cache[key]
        entry.last_use_step = self.step
        return entry

    @staticmethod
    def get_keys(batch_config: dict[str, Tensor], shift: Tensor) -> list[tuple]:
        columns = (
            batch_config["min_noising_strength"].tolist(),
            batch_config["max_noising_strength"].tolist(),
            batch_config["noising_bias"].tolist(),
            batch_config["noising_weight"].tolist(),
            shift.tolist(),
        )
        return [*zip(*columns, strict=True)] # rows
