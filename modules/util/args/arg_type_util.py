import torch


def torch_device(device_name: str) -> torch.device:
    return torch.device(device_name)

