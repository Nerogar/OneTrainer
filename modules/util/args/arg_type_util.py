import torch


def torch_device(device_name: str) -> torch.device:
    return torch.device(device_name)


def torch_dtype(dtype_name: str) -> torch.dtype:
    match dtype_name:
        case 'float32':
            return torch.float32
        case 'float16':
            return torch.float16
        case 'bfloat16':
            return torch.bfloat16
        case _:
            raise NotImplementedError
