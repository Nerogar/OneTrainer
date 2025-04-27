import torch


def get_indices(batch: dict, predicate, device: torch.device, base_element:str = 'image_path'):
    int_indices = [i for i in range(len(batch[base_element]))
                   if predicate({k: v[i] for k, v in batch.items()})]
    return torch.tensor(int_indices, dtype=torch.long, device=device)

def subbatch(batch: dict, indices: list[int], device: torch.device):
    return {
        k: v[indices.to(v.device)] if isinstance(v, torch.Tensor) #.to(): batch['loss_weight'] is on CPU for some reason
                                   else [v[i.item()] for i in indices]
        for k, v in batch.items()
    }
