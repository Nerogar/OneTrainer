import gc
from contextlib import nullcontext

import accelerate
import torch

accelerator = accelerate.Accelerator()
default_device = accelerator.device


def state_dict_has_prefix(state_dict: dict | None, prefix: str):
    if not state_dict:
        return False
    return any(k.startswith(prefix) for k in state_dict)


def to_device_(
        data: torch.Tensor | list | tuple | dict,
        device: torch.device,
        include_parameter_indices: list[int] | None = None,
        non_blocking: bool = False,
):
    if include_parameter_indices is None:
        include_parameter_indices = []

    if isinstance(data, torch.Tensor):
        data.data = data.data.to(device=device, non_blocking=non_blocking)
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if i in include_parameter_indices:
                to_device_(elem, device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        for elem in data.values():
            to_device_(elem, device, non_blocking=non_blocking)


def device_equals(device1: torch.device, device2: torch.device) -> bool:
    return device1.type == device2.type \
        and (0 if device1.index is None else device1.index) == (0 if device1.index is None else device2.index)


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def torch_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def create_stream_context(stream: torch.cuda.Stream) -> torch.cuda.StreamContext | nullcontext:
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.StreamContext(stream)
    return nullcontext()
