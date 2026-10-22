"""
Entropic Time Scheduler (Stancevic et al., "Entropic Time Schedulers for Generative
Diffusion Models", arXiv:2504.13612) - automatic schedule estimation for sigma
(flow-matching) models.

The entropic time
function is estimated from the current model and dataset:

1. For a grid of timesteps, create x_t = (1 - sigma) * x_0 + sigma * epsilon with
   the linear sigma used by ModelSetupFlowMatchingMixin (sigma = t / T).
2. Run the transformer and recover the x_0 prediction: x_0_hat = x_t - sigma * flow.
3. ERROR(sigma) = mean ||x_0_hat - x_0||^2 over the dataset samples.
4. D_entropy = ERROR / sigma^2; entropy = cumsum(D_entropy * d_sigma), rescaled to [0, 1].

The result is cached in <workspace_dir>/entropic/schedule.pt using the same format
as the precomputed repository schedules, so it can also be selected manually in the
UI afterwards:
  {'time': ascending sigma grid including 0, 'time_func': entropy 0..1, 'help': str}
"""

import os

import modules.util.multi_gpu_util as multi
from modules.util.tqdm_util import tqdm

import torch
from torch import Tensor


def entropic_schedule_path(config) -> str:
    return os.path.join(config.workspace_dir, "entropic", "schedule.pt")


def is_supported(model) -> bool:
    """The entropic schedule is only implemented for sigma (flow-matching) models."""
    scheduler = getattr(model, "noise_scheduler", None)
    return (
        scheduler is not None
        and hasattr(scheduler, "timesteps")
        and len(scheduler.timesteps) > 0
        and hasattr(model, "scale_latents")
    )


def _collect_batches(data_loader, num_samples: int) -> list[dict]:
    """Read num_samples latents (plus their conditioning) from the data loader.

    The latents and text encodings are already cached to disk by the data loader,
    so this only reads from the existing cache. Batches are kept on CPU.
    """
    data_loader.get_data_set().start_next_epoch()

    batches = []
    total = 0
    for batch in data_loader.get_data_loader():
        batches.append({
            key: value.cpu() if isinstance(value, Tensor) else value
            for key, value in batch.items()
        })
        total += batch['latent_image'].shape[0]
        if total >= num_samples:
            break

    return batches


def _compute_schedule(model, model_setup, batches: list[dict], config, num_eval_steps: int) -> dict:
    train_device = torch.device(config.train_device)
    num_train_timesteps = model.noise_scheduler.timesteps.shape[0]
    t_values = torch.linspace(1.0, float(num_train_timesteps - 1), num_eval_steps).to(torch.long)

    original_get_timestep = model_setup._get_timestep_discrete
    original_add_noise = model_setup._add_noise_discrete
    captured = {}

    errors = torch.zeros(num_eval_steps, dtype=torch.float64)
    counts = torch.zeros(num_eval_steps, dtype=torch.float64)
    sigmas = torch.zeros(num_eval_steps, dtype=torch.float64)

    try:
        for i, t in enumerate(tqdm(t_values.tolist(), desc="entropic schedule", disable=not multi.is_master())):
            # Pin the timestep for this evaluation and capture the (noisy, sigma) pair.
            def fixed_get_timestep(*args, **kwargs):
                batch_size = args[3] if len(args) > 3 else kwargs.get('batch_size', 1)
                return torch.full((batch_size,), int(t), dtype=torch.long, device=train_device)

            def capturing_add_noise(scaled_latent, latent_noise, timestep, timesteps):
                noisy, sigma = original_add_noise(scaled_latent, latent_noise, timestep, timesteps)
                captured['noisy'] = noisy.detach()
                captured['sigma'] = sigma.detach()
                return noisy, sigma

            model_setup._get_timestep_discrete = fixed_get_timestep
            model_setup._add_noise_discrete = capturing_add_noise

            total_error = 0.0
            total_count = 0
            with torch.no_grad():
                for batch in batches:
                    device_batch = {
                        key: value.to(train_device) if isinstance(value, Tensor) else value
                        for key, value in batch.items()
                    }
                    model_output_data = model_setup.predict(
                        model, device_batch, config, model.train_progress, deterministic=True)
                    x0 = model.scale_latents(device_batch['latent_image'])
                    sigma = captured['sigma'].reshape(-1, 1, 1, 1)
                    x0_hat = captured['noisy'] - sigma * model_output_data['predicted']
                    per_sample_error = ((x0_hat - x0) ** 2).flatten(1).mean(1)
                    total_error += per_sample_error.sum().item()
                    total_count += per_sample_error.shape[0]

            errors[i] = total_error
            counts[i] = total_count
            sigmas[i] = captured['sigma'].mean().item()
    finally:
        model_setup._get_timestep_discrete = original_get_timestep
        model_setup._add_noise_discrete = original_add_noise
        # The pinned-timestep stub skipped the real sampler; clear any cached sampling
        # weights so the real distribution is built from the freshly computed schedule.
        for key in [k for k in model_setup.__dict__ if k.endswith('__weights')]:
            model_setup.__dict__[key] = None

    time = torch.cat([torch.zeros(1, dtype=torch.float64), sigmas])
    density = errors / counts / sigmas.square()
    delta_time = time[1:] - time[:-1]
    entropy = torch.cat([torch.zeros(1, dtype=torch.float64), (density * delta_time).cumsum(0)])
    entropy = entropy - entropy[0]
    entropy = entropy / entropy[-1]

    return {
        'time': time,
        'time_func': entropy,
        'help': (
            "Entropic time function estimated from the model and dataset "
            "(Stancevic et al., arXiv:2504.13612). "
            f"{num_eval_steps} evaluation steps, {int(counts[0])} samples, sigma in [0, 1]. "
            "Delete this file to recompute."
        ),
    }


def load_or_compute_entropic_schedule(
        model,
        model_setup,
        data_loader,
        config,
        num_eval_steps: int = 128,
        num_samples: int = 256,
) -> str | None:
    """Return the path of the entropic schedule file, computing and caching it if needed."""
    path = entropic_schedule_path(config)
    if os.path.isfile(path):
        print(f"Using cached entropic timestep schedule {path}")
        return path

    if not is_supported(model):
        print(
            "Warning: the entropic timestep distribution is only implemented for "
            "sigma (flow-matching) models with a noise scheduler. "
            "Falling back to uniform timestep sampling."
        )
        return None

    # Only the master process computes; the other ranks wait at the barrier.
    for _ in multi.master_first():
        if not multi.is_master():
            continue

        batches = _collect_batches(data_loader, num_samples)
        total = sum(batch['latent_image'].shape[0] for batch in batches)
        print(f"Computing entropic timestep schedule from {total} samples "
              f"({num_eval_steps} evaluation steps), saving to {path}")

        schedule = _compute_schedule(model, model_setup, batches, config, num_eval_steps)
        del batches

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(schedule, path)
        print(f"Saved entropic timestep schedule to {path}")

    return path
