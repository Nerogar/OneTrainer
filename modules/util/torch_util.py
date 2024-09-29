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


def tensor_to_device_(
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
                tensor_to_device_(elem, device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        for elem in data.values():
            tensor_to_device_(elem, device, non_blocking=non_blocking)

def module_to_device_except_sub_module(
        module: torch.nn.Module,
        device: torch.device,
        sub_modules: list[torch.nn.Module],
        non_blocking: bool = False,
):
    sub_module_parameters = set(sum([list(x.parameters()) for x in sub_modules], []))

    def convert(t):
        if t in sub_module_parameters:
            return t

        try:
            return t.to(
                device,
                None,
                non_blocking,
            )
        except NotImplementedError as e:
            if str(e) == "Cannot copy out of meta tensor; no data!":
                raise NotImplementedError(
                    f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                    f"when moving module from meta to a different device."
                ) from None
            else:
                raise

    return module._apply(convert)


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
