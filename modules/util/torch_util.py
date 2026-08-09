import gc
import platform
from collections.abc import Callable
from contextlib import nullcontext

import torch

import accelerate
import packaging
from packaging.version import Version

accelerator = accelerate.Accelerator()
default_device = accelerator.device

torch_version = packaging.version.parse(torch.__version__)


def supports_mem_pool(device: torch.device) -> bool:
    return device.type == "cuda"


def create_mem_pool(device: torch.device):
    # a dedicated MemPool the caller can allocate into; None on devices without MemPool support (cpu/mps)
    return torch.cuda.MemPool() if supports_mem_pool(device) else None


def mem_pool_context(mem_pool):
    # route allocations made in this context into the given MemPool; no-op when it is None
    return torch.cuda.use_mem_pool(mem_pool) if mem_pool is not None else nullcontext()


def state_dict_has_prefix(state_dict: dict | None, prefix: str):
    if not state_dict:
        return False
    # Match on a dot-delimited segment boundary so "text_encoder" does not also match
    # "text_encoder_2". Callers pass the bare prefix; the trailing dot is added here.
    prefix = prefix + '.'
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
        num_bytes = x.numel() * x.element_size()
        err = cudart.cudaHostRegister(
            x.data_ptr(),
            num_bytes,
            0,
        )

        if err.value != 0:
            hint = ""
            if err.value == 1 and num_bytes >= 2**31:
                # the kernel's page list for a registration holds one entry per 4 KiB, so at 2 GiB it reaches
                # 4 MiB, the largest single kmalloc there is, and the pin is refused whatever the driver or
                # GPU. cudaErrorInvalidValue at this size has no other cause, so the attribution is safe.
                hint = (f". A single pinned allocation of {num_bytes / 2**30:.2f} GiB failed: linux "
                        f"{platform.release()} cannot pin 2 GiB or more in one call, a kernel bug present in "
                        f"6.11 and 6.12 and fixed in 6.13. Update the kernel, or run on a host with 6.13 or "
                        f"newer. This attempt leaked {num_bytes / 2**30:.2f} GiB of host memory until reboot")
            raise RuntimeError(f"CUDA Error while trying to pin memory. error: {err.value}, ptr: {x.data_ptr()}, size: {num_bytes}{hint}")


def unpin_tensor_(x):
    # not implemented for other device types
    if torch.cuda.is_available():
        cudart = torch.cuda.cudart()
        err = cudart.cudaHostUnregister(x.data_ptr())

        if err.value != 0:
            raise RuntimeError(f"CUDA Error while trying to unpin memory. error {err.value}, ptr: {x.data_ptr()}")
