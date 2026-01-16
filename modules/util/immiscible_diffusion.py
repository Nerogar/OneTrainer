import torch


def immiscible_oversampling(self, source_tensor, noise_candidates):
    """
    Implements the noise oversampling method proposed in the paper:
    "Improved Immiscible Diffusion: Accelerate Diffusion Training by Reducing Its Miscibility"
    (https://arxiv.org/abs/2505.18521)
    Modified from:
    https://github.com/yhli123/Immiscible-Diffusion/blob/65feadc66c7653b2644d06246fe8a424a5d76794/KNN-stable_diffusion/conditional_ft_train_sd.py#L940
    """
    # Using float16 for distance calculation to save memory/compute as per reference implementation
    latents_points = source_tensor.flatten(start_dim=1).to(torch.float16)
    noise_points = noise_candidates.flatten(start_dim=2).to(torch.float16)

    # Calculate L2 distance between latents and the corresponding k noises
    distance_points = latents_points.unsqueeze(1) - noise_points

    # Euclidean norm
    distance = torch.linalg.vector_norm(distance_points, dim=2) # [B, k]

    # Pick the nearest noise index for each data point
    _, min_index = torch.min(distance, dim=1)

    expand_shape = [-1, 1] + [source_tensor.shape[i] for i in range(1, source_tensor.ndim)]
    gather_index = min_index.view(-1, *([1] * (source_tensor.ndim))).expand(*expand_shape)
    selected_noise = torch.gather(noise_candidates, 1, gather_index)

    # Squeeze the 'k' dimension (which is now size 1) to return to [B, C, H, W]
    noise = selected_noise.squeeze(1)
    return noise
