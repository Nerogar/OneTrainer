import copy
import math

from modules.util.config.SampleConfig import parse_override_shift

import torch

from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler

import numpy as np

FlowScheduler = FlowMatchEulerDiscreteScheduler | DPMSolverMultistepScheduler


def prepare_flow_match_scheduler(
        scheduler: FlowScheduler,
        steps: int,
        device: torch.device,
        override_shift: float | None,
        *,
        mu: float | None = None,
        sigmas: np.ndarray | None = None,
) -> FlowScheduler:
    """Build one sample's schedule without mutating the model scheduler."""
    shift = parse_override_shift(override_shift)
    if shift is not None and isinstance(scheduler, DPMSolverMultistepScheduler):
        other_schedules = ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "use_lu_lambdas")
        if not scheduler.config.use_flow_sigmas or any(scheduler.config.get(option) for option in other_schedules):
            raise ValueError("Sampling shift requires a multistep scheduler configured to use flow sigmas.")

    if shift is None:
        scheduler = copy.deepcopy(scheduler)
    elif scheduler.config.use_dynamic_shifting:
        scheduler = copy.deepcopy(scheduler)
        mu = math.log(shift) if scheduler.config.time_shift_type == "exponential" else shift
    elif isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        # Euler derives sigma_min from the shift at construction. set_shift() leaves it stale.
        scheduler = scheduler.from_config(scheduler.config, shift=shift)
    else:
        scheduler = scheduler.from_config(scheduler.config, flow_shift=shift)

    schedule_args = {}
    if mu is not None:
        schedule_args["mu"] = mu
    if sigmas is not None:
        schedule_args["sigmas"] = sigmas
    scheduler.set_timesteps(steps, device=device, **schedule_args)
    return scheduler
