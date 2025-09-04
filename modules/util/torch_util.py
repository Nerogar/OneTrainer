import gc
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch

import accelerate
import packaging
from packaging.version import Version

accelerator = accelerate.Accelerator()
default_device = accelerator.device

torch_version = packaging.version.parse(torch.__version__)


def state_dict_has_prefix(state_dict: dict | None, prefix: str):
    if not state_dict:
        return False
    return any(k.startswith(prefix) for k in state_dict)

def get_tensor_data(
        data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    tensors = []

    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        return [data.data]
    elif isinstance(data, list | tuple):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                tensors.extend(get_tensor_data(elem))
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            tensors.extend(get_tensor_data(elem))

    return tensors


def has_grad_fn(
        data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
) -> bool:
    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        return data.grad_fn is not None
    elif isinstance(data, list | tuple):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                if has_grad_fn(elem):
                    return True
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            if has_grad_fn(elem):
                return True

    return False

def add_dummy_grad_fn_(
        data: torch.Tensor | list | tuple | dict,
) -> Any:
    if isinstance(data, torch.Tensor):
        if data.grad_fn is not None:
            return data
        grad_tensor = torch\
            .zeros(size=(0, *data.shape[1:]), requires_grad=True, device=data.device, dtype=data.dtype)
        return torch.cat([data, grad_tensor], dim=0)
    if isinstance(data, list):
        for i, elem in enumerate(data):
            if isinstance(elem, torch.Tensor):
                if elem.grad_fn is not None:
                    return data
                grad_tensor = torch\
                    .zeros(size=(0, *elem.shape[1:]), requires_grad=True, device=elem.device, dtype=elem.dtype)
                data[i] = torch.cat([elem, grad_tensor], dim=0)
                return data
            else:
                data[i] = add_dummy_grad_fn_(elem)
    if isinstance(data, tuple):
        for i, elem in enumerate(data):
            if isinstance(elem, torch.Tensor):
                if elem.grad_fn is not None:
                    return data
                grad_tensor = torch\
                    .zeros(size=(0, *elem.shape[1:]), requires_grad=True, device=elem.device, dtype=elem.dtype)
                data = list(data)
                data[i] = torch.cat([elem, grad_tensor], dim=0)
                data = tuple(data)
                return data
            else:
                data = list(data)
                data[i] = add_dummy_grad_fn_(elem)
                data = tuple(data)
    elif isinstance(data, dict):
        for key, elem in data.items():
            if isinstance(elem, torch.Tensor):
                if elem.grad_fn is not None:
                    return data
                grad_tensor = torch \
                    .zeros(size=(0, *elem.shape[1:]), requires_grad=True, device=elem.device, dtype=elem.dtype)
                data[key] = torch.cat([elem, grad_tensor], dim=0)
                return data
            else:
                data[key] = add_dummy_grad_fn_(elem)

    return data

def tensors_to_device_(
        data: torch.Tensor | list | tuple | dict,
        device: torch.device,
        include_parameter_indices: list[int] | None = None,
        non_blocking: bool = False,
        allocator: Callable[[torch.tensor], torch.tensor] | None = None,
) -> bool:
    tensor_transferred = False

    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        if allocator is None:
            data.data = data.data.to(device=device, non_blocking=non_blocking)
        else:
            tensor = allocator(data)
            tensor.copy_(data, non_blocking=non_blocking)
            data.data = tensor
        tensor_transferred = True
    elif isinstance(data, list | tuple):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                tensor_transferred |= tensors_to_device_(elem, device, non_blocking=non_blocking, allocator=allocator)
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            tensor_transferred |= tensors_to_device_(elem, device, non_blocking=non_blocking, allocator=allocator)

    return tensor_transferred


def optimizer_to_device_(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state_dict()['state'].values():
        tensors_to_device_(state, device)


def replace_tensors_(
        target_data: torch.Tensor | list | tuple | dict,
        source_data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
):
    if isinstance(target_data, torch.Tensor) and include_parameter_indices is None:
        target_data.data = source_data.data
    elif isinstance(target_data, list | tuple):
        for i, elem in enumerate(target_data):
            if include_parameter_indices is None or i in include_parameter_indices:
                replace_tensors_(elem, source_data[i])
    elif isinstance(target_data, dict) and include_parameter_indices is None:
        for key, elem in target_data.items():
            replace_tensors_(elem, source_data[key])


def tensors_match_device(
        data: torch.Tensor | list | tuple | dict,
        device: torch.device,
        include_parameter_indices: list[int] | None = None,
) -> bool:
    if isinstance(data, torch.Tensor) and include_parameter_indices is None:
        if not device_equals(data.device, device):
            return False
    elif isinstance(data, list | tuple):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                if not tensors_match_device(elem, device):
                    return False
    elif isinstance(data, dict) and include_parameter_indices is None:
        for elem in data.values():
            if not tensors_match_device(elem, device):
                return False

    return True


def tensors_record_stream(
        stream: torch.Stream,
        data: torch.Tensor | list | tuple | dict,
        include_parameter_indices: list[int] | None = None,
):
    if isinstance(data, torch.Tensor):
        if data.device.type == "cuda":
            data.record_stream(stream)
    elif isinstance(data, list | tuple):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                # [] intentional - process all tensors inside the selected parameter(s)
                tensors_record_stream(stream, elem, [])
    elif isinstance(data, dict):
        for elem in data.values():
            tensors_record_stream(stream, elem)


def unpin_module(
        module: torch.nn.Module,
):
    def convert(t):
        if t.is_pinned():
            return t.clone()
        return t

    return module._apply(convert)


def device_equals(device1: torch.device, device2: torch.device) -> bool:
    return device1 is not None and device2 is not None \
        and device1.type == device2.type \
        and (0 if device1.index is None else device1.index) == (0 if device2.index is None else device2.index)


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        if torch_version > Version("2.6.0"):
            # TODO: replace with a torch.cuda binding once that's available
            torch._C._host_emptyCache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def torch_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        packaging.version.parse(torch.__version__)


def create_stream_context(stream: torch.cuda.Stream) -> torch.cuda.StreamContext | nullcontext:
    if isinstance(stream, torch.cuda.Stream):
        return torch.cuda.StreamContext(stream)
    return nullcontext()


def pin_tensor_(x):
    # not implemented for other device types
    if torch.cuda.is_available():
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostRegister(
            x.data_ptr(),
            x.numel() * x.element_size(),
            0,
        )

        if err.value != 0:
            raise RuntimeError(f"CUDA Error while trying to pin memory. error: {err.value}, ptr: {x.data_ptr()}, size: {x.numel() * x.element_size()}")


def unpin_tensor_(x):
    # not implemented for other device types
    if torch.cuda.is_available():
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostUnregister(x.data_ptr())

        if err.value != 0:
            raise RuntimeError(f"CUDA Error while trying to unpin memory. error {err.value}, ptr: {x.data_ptr()}")
