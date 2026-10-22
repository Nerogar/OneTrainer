"""
Entropic Time Scheduler (Stancevic et al., "Entropic Time Schedulers for Generative
Diffusion Models", arXiv:2504.13612) - automatic schedule estimation for sigma
(flow-matching) models.

The entropic time function is estimated from the current model and dataset:

1. For a grid of timesteps, create x_t = (1 - sigma) * x_0 + sigma * epsilon with
   the linear sigma used by ModelSetupFlowMatchingMixin (sigma = t / T).
2. Run the transformer and recover the x_0 prediction: x_0_hat = x_t - sigma * flow.
3. ERROR(sigma) = mean ||x_0_hat - x_0||^2 over the dataset samples.
4. D_entropy = ERROR / sigma^2; entropy = cumsum(D_entropy * d_sigma), rescaled to [0, 1].

The result is cached in <workspace_dir>/entropic/schedule.pt using the same format
as the referenced paper, so it can also be selected manually in the
UI afterwards:
  {'time': ascending sigma grid including 0, 'time_func': entropy 0..1, 'help': str}

Performance note: instead of pinning one timestep per pass (num_eval_steps x
num_batches tiny forward passes), timesteps are processed in chunks that are tiled
into the batch. The pipeline already supports per-sample timesteps, so one forward
pass evaluates timestep_chunk_size x batch_size samples at once. All error/sigma
reductions stay on the train device with index_add_.

Additionally, latents are downscaled once at collection time so their total
image-space area is about schedule_resolution x schedule_resolution pixels
(default None, aspect ratio preserved, area interpolation). Because the entropy
curve is normalized to [0, 1], its shape is well preserved at reduced resolution;
this is an approximation that trades a little accuracy for a large speedup
(fewer transformer tokens per sample). Pass schedule_resolution=None to evaluate
at the native training resolution instead.
"""

import math
import os

import modules.util.multi_gpu_util as multi
from modules.util.tqdm_util import tqdm

import torch
import torch.nn.functional as F


def _to_device(data, device):
    """Recursively move tensors in nested dicts/lists/tuples to the given device."""
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_device(v, device) for v in data)
    return data


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


def _latent_downscale_factor(model) -> int:
    """Best-effort detection of the VAE spatial downscale factor (8x for most models)."""
    vae = getattr(model, "vae", None)
    config = getattr(vae, "config", None)
    block_out_channels = getattr(config, "block_out_channels", None)
    if block_out_channels and len(block_out_channels) >= 2:
        return 2 ** (len(block_out_channels) - 1)
    print("Warning: could not detect the VAE downscale factor, assuming 8x.")
    return 8


def _downscale_batch(batch: dict, schedule_resolution: float, downscale: int, first_resize: list) -> dict:
    """Downscale all latent-sized spatial tensors in a batch so that the image-space
    area is about schedule_resolution x schedule_resolution pixels (aspect preserved).

    Only downscaling is applied: batches that are already at or below the target
    resolution are returned unchanged. Floating point tensors (latents) use area
    interpolation (average pooling preserves local statistics when downsampling);
    other tensors (masks and index maps) use nearest interpolation so their values
    stay valid.
    """
    if not schedule_resolution or schedule_resolution <= 0:
        return batch

    latent = batch['latent_image']
    if not torch.is_tensor(latent) or latent.ndim < 4:
        return batch

    height, width = latent.shape[-2], latent.shape[-1]
    image_area = (height * downscale) * (width * downscale)
    target_area = schedule_resolution * schedule_resolution
    if image_area <= target_area:
        return batch

    scale = math.sqrt(target_area / image_area)
    target_height = max(8, round(height * scale / 8) * 8)
    target_width = max(8, round(width * scale / 8) * 8)
    if target_height == height and target_width == width:
        return batch

    def resize(value):
        if not (torch.is_tensor(value) and value.ndim >= 4
                and value.shape[-2] == height and value.shape[-1] == width):
            return value
        original_dtype = value.dtype
        mode = 'area' if value.is_floating_point() else 'nearest'
        value = F.interpolate(value.float(), size=(target_height, target_width), mode=mode)
        return value.to(original_dtype)

    if not first_resize[0]:
        print(f"Downscaling schedule latents to {target_height}x{target_width} "
              f"(~{int(schedule_resolution)}x{int(schedule_resolution)} pixels "
              f"at {downscale}x VAE downscale)")
        first_resize[0] = True

    return {key: resize(value) for key, value in batch.items()}


def _collect_batches(data_loader, num_samples: int, schedule_resolution: float | None = None,
                     downscale: int = 8) -> list[dict]:
    """Read num_samples latents (plus their conditioning) from the data loader."""
    data_loader.get_data_set().start_next_epoch()

    batches = []
    first_resize = [False]
    total = 0
    for batch in data_loader.get_data_loader():
        batch = _downscale_batch(batch, schedule_resolution, downscale, first_resize)
        # Offload batch to CPU memory
        batch = _to_device(batch, "cpu")
        batches.append(batch)
        total += batch['latent_image'].shape[0]
        if total >= num_samples:
            break

    return batches


def _tile_batch(batch: dict, k: int) -> dict:
    """Repeat every entry of a batch k times along the batch dimension."""
    batch_size = batch['latent_image'].shape[0]

    def tile(value):
        if torch.is_tensor(value):
            if value.ndim >= 1 and value.shape[0] == batch_size:
                return value.repeat_interleave(k, dim=0)
            return value
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            return list(value) * k
        return value

    return {key: tile(value) for key, value in batch.items()}


def _compute_schedule(
        model,
        model_setup,
        batches: list[dict],
        config,
        num_eval_steps: int,
        timestep_chunk_size: int = 8,
        schedule_resolution: float | None = None,
) -> dict:
    train_device = torch.device(config.train_device)
    num_train_timesteps = model.noise_scheduler.timesteps.shape[0]
    t_values = torch.linspace(1.0, float(num_train_timesteps - 1), num_eval_steps).to(torch.long)

    original_get_timestep = model_setup._get_timestep_discrete
    original_add_noise = model_setup._add_noise_discrete
    captured = {}

    errors = torch.zeros(num_eval_steps, dtype=torch.float64, device=train_device)
    counts = torch.zeros(num_eval_steps, dtype=torch.float64, device=train_device)
    sigma_sums = torch.zeros(num_eval_steps, dtype=torch.float64, device=train_device)

    progress = tqdm(
        total=len(batches) * math.ceil(num_eval_steps / timestep_chunk_size),
        desc="entropic schedule",
        disable=not multi.is_master(),
    )
    try:
        with torch.inference_mode(), torch.autocast(device_type=train_device.type, dtype=torch.bfloat16):
            for batch in batches:
                # Move only the current batch to the GPU
                gpu_batch = _to_device(batch, train_device)
                x0 = model.scale_latents(gpu_batch['latent_image'])
                batch_size = x0.shape[0]

                for chunk_start in range(0, num_eval_steps, timestep_chunk_size):
                    t_chunk = t_values[chunk_start:chunk_start + timestep_chunk_size]
                    k = t_chunk.shape[0]

                    # Sample i of the tiled batch is original sample i // k, copy i % k,
                    # so its timestep is t_chunk[i % k] and its schedule index is
                    # chunk_start + (i % k).
                    t_for_sample = t_chunk.repeat(batch_size)
                    t_index = torch.arange(chunk_start, chunk_start + k, device=train_device) \
                        .repeat(batch_size)

                    def fixed_get_timestep(*args, t_for_sample=t_for_sample, **kwargs):
                         return t_for_sample

                    def capturing_add_noise(scaled_latent, latent_noise, timestep, timesteps):
                        noisy, sigma = original_add_noise(scaled_latent, latent_noise, timestep, timesteps)
                        captured['noisy'] = noisy.detach()
                        captured['sigma'] = sigma.detach().reshape(-1)
                        return noisy, sigma

                    model_setup._get_timestep_discrete = fixed_get_timestep
                    model_setup._add_noise_discrete = capturing_add_noise

                    tiled_batch = _tile_batch(gpu_batch, k)
                    model_output_data = model_setup.predict(
                        model, tiled_batch, config, model.train_progress, deterministic=True)
                    del tiled_batch

                    x0_hat = captured['noisy'] \
                        - captured['sigma'].reshape(-1, 1, 1, 1) * model_output_data['predicted']
                    per_sample_error = (x0_hat - x0.repeat_interleave(k, dim=0)) \
                        .square().flatten(1).mean(1).double()

                    errors.index_add_(0, t_index, per_sample_error)
                    counts.index_add_(0, t_index, torch.ones_like(per_sample_error))
                    sigma_sums.index_add_(0, t_index, captured['sigma'].double())
                    progress.update(1)

                    del model_output_data

                # Release the GPU memory for this batch before loading the next
                del gpu_batch, x0
                captured.clear()
    finally:
        captured.clear()
        model_setup._get_timestep_discrete = original_get_timestep
        model_setup._add_noise_discrete = original_add_noise
        # The pinned-timestep stub skipped the real sampler; clear any cached sampling
        # weights so the real distribution is built from the freshly computed schedule.
        for key in [key for key in model_setup.__dict__ if key.endswith('__weights')]:
            model_setup.__dict__[key] = None
        progress.close()

    errors = errors.cpu()
    counts = counts.cpu()
    sigmas = sigma_sums.cpu() / counts.clamp_min(1.0)

    time = torch.cat([torch.zeros(1, dtype=torch.float64), sigmas])
    density = errors / counts / sigmas.square()
    delta_time = time[1:] - time[:-1]
    entropy = torch.cat([torch.zeros(1, dtype=torch.float64), (density * delta_time).cumsum(0)])
    entropy = entropy - entropy[0]
    entropy = entropy / entropy[-1]

    help_text = (
        "Entropic time function estimated from the model and dataset "
        "(Stancevic et al., arXiv:2504.13612). "
        f"{num_eval_steps} evaluation steps, {int(counts[0])} samples, sigma in [0, 1]. "
    )
    if schedule_resolution:
        help_text += (
            f"Estimated at reduced resolution (~{int(schedule_resolution)}x{int(schedule_resolution)} "
            "image pixels, area-interpolated latents); the normalized curve is an approximation. "
        )
    help_text += "Delete this file to recompute."

    return {
        'time': time,
        'time_func': entropy,
        'help': help_text,
    }


def load_or_compute_entropic_schedule(
        model,
        model_setup,
        data_loader,
        config,
        num_eval_steps: int = 8,
        num_samples: int = 32,
        timestep_chunk_size: int = 8,
        schedule_resolution: int | None = None,
) -> str | None:
    """Return the path of the entropic schedule file, computing and caching it if needed.

    schedule_resolution: if set, latents are downscaled so their total image-space
    area is about schedule_resolution x schedule_resolution pixels (aspect ratio
    preserved) before the schedule is estimated. Pass None to evaluate at the
    native training resolution.
    """
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

    downscale = _latent_downscale_factor(model)

    # Only the master process computes; the other ranks wait at the barrier.
    for _ in multi.master_first():
        if not multi.is_master():
            continue

        batches = _collect_batches(data_loader, num_samples, schedule_resolution, downscale)
        total = sum(batch['latent_image'].shape[0] for batch in batches)
        print(f"Computing entropic timestep schedule from {total} samples "
              f"({num_eval_steps} evaluation steps, timestep chunk size {timestep_chunk_size}, "
              f"schedule resolution {int(schedule_resolution) if schedule_resolution else 'native'}), "
              f"saving to {path}")

        schedule = _compute_schedule(
            model, model_setup, batches, config, num_eval_steps, timestep_chunk_size,
            schedule_resolution)
        del batches

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(schedule, path)
        print(f"Saved entropic timestep schedule to {path}")

    return path
