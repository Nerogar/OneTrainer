from collections.abc import Callable

import torch


def get_indices(
        batch: dict,
        batch_size: int,
        predicate: Callable[[int, dict], bool],
) -> list[int]:
    return [i for i in range(batch_size) if predicate(i, batch)]

def subbatch(batch: dict, indices: list[int], device: torch.device):
    return {
        k: v[indices.to(v.device)] if isinstance(v, torch.Tensor) #.to(): batch['loss_weight'] is on CPU for some reason
                                   else [v[i.item()] for i in indices]
        for k, v in batch.items()
    }
