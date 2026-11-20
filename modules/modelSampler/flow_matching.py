import copy
import math

from modules.util.config.SampleConfig import parse_override_shift

import torch

from diffusers import FlowMatchEulerDiscreteScheduler

import numpy as np


def prepare_flow_match_scheduler(
        scheduler: FlowMatchEulerDiscreteScheduler,
        steps: int,
        device: torch.device,
        override_shift: float | None,
        *,
        mu: float | None = None,
        sigmas: np.ndarray | None = None,
) -> FlowMatchEulerDiscreteScheduler:
    """Build one sample's schedule without mutating the model scheduler."""
    shift = parse_override_shift(override_shift)
    if shift is None:
        scheduler = copy.deepcopy(scheduler)
    elif scheduler.config.use_dynamic_shifting:
        scheduler = copy.deepcopy(scheduler)
        mu = math.log(shift) if scheduler.config.time_shift_type == "exponential" else shift
    else:
        # Euler derives sigma_min from the shift at construction. set_shift() leaves it stale.
        scheduler = scheduler.from_config(scheduler.config, shift=shift)

    scheduler.set_timesteps(steps, device=device, sigmas=sigmas, mu=mu)
    return scheduler
