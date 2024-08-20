import torch


def torch_device(device_name: str) -> torch.device:
    return torch.device(device_name)

def nullable_bool(bool_value: str) -> bool:
    return bool_value.lower() == "true"
