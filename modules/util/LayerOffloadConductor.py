import math
import random
from collections import deque
from typing import Any

from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.quantization_util import get_offload_tensor_bytes, offload_quantized
from modules.util.torch_util import (
    create_stream_context,
    device_equals,
    pin_tensor_,
    tensors_record_stream,
    torch_gc,
    unpin_tensor_,
)

import torch
from torch import nn

MESSAGES = []


# Only relevant with activation offloading. Each layer-call the CPU enqueues ahead of the GPU floats one
# layer's worth of activations - the forward's copy stays alive until its D2H runs, the backward's reload
# destination is allocated when the CPU reaches the block - so the run-ahead has to be capped for peak VRAM
# to be predictable at all. Saturation needs only enough queued work to cover CPU-side jitter (tens of ms)
# against a layer-call of tens to hundreds of ms, so a small cap costs no throughput. 0 = unbounded.
#
# Also the number of trailing layers whose activations are not offloaded: the backward consumes N-1..0, so
# the offloads still draining when the forward ends - at most this many - are the ones needed first. Keeping
# those layers resident is free at this cap and saves a D2H/H2D round-trip of the same bytes.
MAX_LAYER_CALLS_IN_FLIGHT = 2


def log(msg: str = ''):
    pass
    # print(msg)
    # MESSAGES.append(msg)


def flat_storage_view(tensor: torch.Tensor) -> torch.Tensor | None:
    # Contiguous 1-D view over a tensor's whole storage region, or None if the strides leave gaps so that the
    # elements between the first and last are not all part of this tensor. A permuted view of a freshly
    # allocated tensor (a transpose, a head-split) is dense and yields a view; a slice of something larger
    # does not. Copying through this view moves the bytes as-is instead of gathering them element by element.
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    if span != tensor.numel():
        return None
    return torch.as_strided(tensor, (tensor.numel(),), (1,), tensor.storage_offset())


def clone_tensor_allocator(tensor: torch.Tensor) -> torch.Tensor:
    # clones a tensor into a new memory location to remove all memory dependencies between tensors
    return tensor.clone()


def ceil_16(number: int) -> int:
    return number + (16 - (number % 16)) % 16


def floor_16(number: int) -> int:
    return number - (number % 16)


class StaticLayerTensorAllocator:
    def __init__(
            self,
            layer_allocator: 'StaticLayerAllocator',
            allocate_forward: bool,
            layer_index: int,
    ):
        self.__layer_allocator = layer_allocator
        self.__allocate_forward = allocate_forward
        self.__layer_index = layer_index

        if allocate_forward:
            self.__allocation_start = layer_allocator.allocation_end
            self.__allocation_end = layer_allocator.allocation_end
            log(f"{self.__layer_allocator.device}/allocating layer {self.__layer_index}, allocation_start {self.__allocation_end:_}")
        else:
            self.__allocation_start = layer_allocator.allocation_start
            self.__allocation_end = layer_allocator.allocation_start
            log(f"{self.__layer_allocator.device}/allocating layer {self.__layer_index}, allocation_end {self.__allocation_start:_}")

    def allocate_like(self, source_tensor: torch.Tensor) -> torch.Tensor:
        num_bytes = source_tensor.numel() * source_tensor.element_size()

        cache_tensor_size = self.__layer_allocator.cache_tensor_size
        total_cache_bytes = cache_tensor_size * len(self.__layer_allocator.cache_tensors)
        if self.__allocate_forward:
            cache_tensor_index = self.__allocation_end // cache_tensor_size
            cache_tensor_allocation_end = ceil_16(self.__allocation_end % cache_tensor_size)

            if cache_tensor_allocation_end + num_bytes > cache_tensor_size:
                # move to the start of the next cache tensor
                cache_tensor_index += 1
                cache_tensor_allocation_end = 0
            if cache_tensor_index * cache_tensor_size + cache_tensor_allocation_end + num_bytes > total_cache_bytes:
                # move to the first cache tensor
                cache_tensor_index = 0
                cache_tensor_allocation_end = 0

            self.__allocation_end = cache_tensor_index * cache_tensor_size + cache_tensor_allocation_end
            self.__layer_allocator.ensure_allocation(cache_tensor_index)
            cache_tensor = self.__layer_allocator.cache_tensors[cache_tensor_index]
            allocated_tensor = cache_tensor[cache_tensor_allocation_end:cache_tensor_allocation_end + num_bytes]
            # log(f"--allocated: {self.__layer_allocator.cache_tensors[cache_tensor_index].device}/{num_bytes} bytes, between {self.__allocation_end} - {self.__allocation_end + num_bytes}/[{cache_tensor_index}] {cache_tensor_allocation_end} for layer {self.__layer_index}")
            self.__allocation_end += num_bytes
            self.__layer_allocator.allocation_end = self.__allocation_end
        else:
            cache_tensor_index = self.__allocation_start // cache_tensor_size
            cache_tensor_allocation_start = self.__allocation_start % cache_tensor_size

            if cache_tensor_allocation_start - num_bytes < 0:
                # move to the end of the previous cache tensor
                cache_tensor_index -= 1
                cache_tensor_allocation_start = cache_tensor_size
            if cache_tensor_index < 0:
                # move to the first cache tensor
                cache_tensor_index = len(self.__layer_allocator.cache_tensors) - 1
                cache_tensor_allocation_start = cache_tensor_size

            new_allocation_start = floor_16(cache_tensor_allocation_start - num_bytes)
            self.__layer_allocator.ensure_allocation(cache_tensor_index)
            cache_tensor = self.__layer_allocator.cache_tensors[cache_tensor_index]
            allocated_tensor = cache_tensor[new_allocation_start:new_allocation_start + num_bytes]
            self.__allocation_start = cache_tensor_index * cache_tensor_size + new_allocation_start
            # log(f"--allocated: {self.__layer_allocator.cache_tensors[cache_tensor_index].device}/{num_bytes} bytes, between {self.__allocation_start - num_bytes} - {self.__allocation_start}/[{cache_tensor_index}] {cache_tensor_allocation_start - num_bytes} for layer {self.__layer_index}")
            self.__layer_allocator.allocation_start = self.__allocation_start

        return allocated_tensor.view(dtype=source_tensor.dtype).view(size=source_tensor.shape)

    def deallocate(self, deallocate_forward):
        if deallocate_forward:
            log(f"{self.__layer_allocator.device}/deallocating layer {self.__layer_index}, allocation_start {self.__allocation_end:_}")
            self.__layer_allocator.allocation_start = self.__allocation_end
        else:
            log(f"{self.__layer_allocator.device}/deallocating layer {self.__layer_index}, allocation_end {self.__allocation_start:_}")
            self.__layer_allocator.allocation_end = self.__allocation_start


class StaticLayerAllocator:
    device: torch.device
    __is_pinned: bool

    __num_layers: int
    __max_tensor_bytes: int
    __layer_bytes: list[int]
    cache_tensors: list[torch.Tensor | None]
    cache_tensor_size: int

    allocation_start: int  # index of the first allocated byte
    allocation_end: int  # index of the first unallocated byte

    __tensor_allocators: list[StaticLayerTensorAllocator | None]

    def __init__(
            self,
            device: torch.device,
    ):
        self.device = device
        self.__allocate_statically = True
        self.__is_pinned = device.type == "cpu"

        self.__num_layers = 0
        self.__max_tensor_bytes = 0
        self.__layer_bytes = []
        self.cache_tensors = []
        self.cache_tensor_size = 0

        self.allocation_start = 0
        self.allocation_end = 0

        self.__tensor_allocators = []

    def allocate_cache(self, layers: list[nn.Module], target_bytes: int):
        if not self.__allocate_statically or any(x is not None for x in self.cache_tensors):
            return

        log(f"allocating cache on device {self.device}")

        self.__max_tensor_bytes = 0
        self.__layer_bytes = []
        for layer in layers:
            layer_tensor_bytes = [get_offload_tensor_bytes(x) for x in layer.modules()]
            self.__max_tensor_bytes = max(self.__max_tensor_bytes, *layer_tensor_bytes)
            self.__layer_bytes.append(sum(layer_tensor_bytes))

        cache_bytes = target_bytes
        num_cache_tensors = min(
            # no more than 10% overhead
            math.ceil(int(cache_bytes * 0.10) / self.__max_tensor_bytes),
            # at least twice self.__max_tensor_bytes for each tensor
            math.ceil(cache_bytes / (self.__max_tensor_bytes * 2)),
            # no more than 10 cache tensors
            10
        )
        # add self.__max_tensor_bytes to ensure even the largest tensors can be allocated in the remaining space
        # add 4kb for the alignment overhead
        self.cache_tensor_size = math.ceil(cache_bytes / num_cache_tensors) + self.__max_tensor_bytes + 4096

        self.__tensor_allocators = [None] * len(layers)
        self.cache_tensors = [None] * num_cache_tensors
        self.allocation_start = 0
        self.allocation_end = 0

    def ensure_allocation(self, cache_tensor_index: int):
        if self.cache_tensors[cache_tensor_index] is None:
            torch_gc()

            self.cache_tensors[cache_tensor_index] = \
                torch.zeros((self.cache_tensor_size,), dtype=torch.int8, device=self.device)

            log(f"tensor {cache_tensor_index} not allocated, allocating {self.cache_tensor_size} bytes")

            if self.__is_pinned:
                pin_tensor_(self.cache_tensors[cache_tensor_index])

    def deallocate_cache(self):
        if not self.__allocate_statically:
            return

        for cache_tensor in self.cache_tensors:
            if cache_tensor is not None and self.__is_pinned:
                unpin_tensor_(cache_tensor)

        self.cache_tensors = [None] * len(self.cache_tensors)
        self.__tensor_allocators = [None] * len(self.__tensor_allocators)

    def get_allocator(self, layer_index: int, allocate_forward: bool) -> StaticLayerTensorAllocator | None:
        if self.__allocate_statically:
            allocator = StaticLayerTensorAllocator(self, allocate_forward, layer_index)
            self.__tensor_allocators[layer_index] = allocator
            return allocator
        else:
            return None

    def deallocate_layer(self, layer_index: int, deallocate_forward: bool):
        if self.__tensor_allocators[layer_index] is not None:
            self.__tensor_allocators[layer_index].deallocate(deallocate_forward)
            self.__tensor_allocators[layer_index] = None


class StaticActivationAllocator:
    __device: torch.device
    __allocate_statically: bool
    __is_pinned: bool

    __cache_tensors: list[torch.Tensor]
    __current_cache_tensor: int
    __current_cache_tensor_offset: int
    __allocated_bytes: int
    __max_allocated_bytes: int

    def __init__(
            self,
            device: torch.device,
    ):
        self.__device = device
        self.__allocate_statically = True
        self.__is_pinned = device.type == "cpu"

        self.__cache_tensors = []
        self.__current_cache_tensor = 0
        self.__current_cache_tensor_offset = 0
        self.__allocated_bytes = 0
        self.__max_allocated_bytes = 0

    @property
    def allocated_bytes(self) -> int:
        return self.__allocated_bytes

    def reserve_cache(self, tensors: list[torch.Tensor]):
        num_bytes = sum(tensor.element_size() * tensor.numel() for tensor in tensors) \
                    + len(tensors) * 16  # add enough padding for alignment

        if num_bytes == 0:
            return

        if len(self.__cache_tensors) == 0:
            num_bytes = max(num_bytes, self.__max_allocated_bytes)

        cache_found = False
        while self.__current_cache_tensor < len(self.__cache_tensors):
            if self.__cache_tensors[self.__current_cache_tensor].shape[0] - self.__current_cache_tensor_offset \
                    >= num_bytes:
                cache_found = True
                break

            self.__current_cache_tensor += 1
            self.__current_cache_tensor_offset = 0

        if not cache_found:
            torch_gc()
            cache_tensor = torch.zeros((num_bytes,), dtype=torch.int8, device=self.__device)
            log(f"{self.__device}/allocating activations cache {num_bytes:_}, total: {self.__allocated_bytes:_}, max: {self.__max_allocated_bytes:_}")

            if self.__is_pinned:
                pin_tensor_(cache_tensor)

            self.__cache_tensors.append(cache_tensor)
            self.__allocated_bytes += num_bytes

        self.__max_allocated_bytes = max(self.__max_allocated_bytes, self.__allocated_bytes)

    def allocate_like(self, source_tensor: torch.Tensor) -> torch.Tensor:
        num_bytes = source_tensor.element_size() * source_tensor.numel()
        cache_tensor = self.__cache_tensors[self.__current_cache_tensor]
        allocated_tensor = \
            cache_tensor[self.__current_cache_tensor_offset:self.__current_cache_tensor_offset + num_bytes]
        self.__current_cache_tensor_offset += ceil_16(num_bytes)

        return allocated_tensor.view(dtype=source_tensor.dtype).view(size=source_tensor.shape)

    def deallocate(self):
        if len(self.__cache_tensors) > 1:
            # more than one tensor was allocated. this can be condensed into a single tensor to reduce fragmentation
            if self.__is_pinned:
                for cache_tensor in self.__cache_tensors:
                    unpin_tensor_(cache_tensor)

            self.__cache_tensors = []
            torch_gc()

            # add 4kb for the alignment overhead
            num_bytes = self.__allocated_bytes + 4096
            cache_tensor = torch.zeros((num_bytes,), dtype=torch.int8, device=self.__device)
            log(f"{self.__device}/condensing activations cache {num_bytes:_}, total: {self.__allocated_bytes:_}, max: {self.__max_allocated_bytes:_}")

            if self.__is_pinned:
                pin_tensor_(cache_tensor)

            self.__cache_tensors = [cache_tensor]

        self.__current_cache_tensor = 0
        self.__current_cache_tensor_offset = 0
        self.__allocated_bytes = sum(cache_tensor.shape[0] for cache_tensor in self.__cache_tensors)

    def deallocate_cache(self):
        if self.__is_pinned:
            for cache_tensor in self.__cache_tensors:
                unpin_tensor_(cache_tensor)

        self.__cache_tensors = []


class SyncEvent:
    def __init__(
            self,
            torch_event: torch.cuda.Event | torch.mps.Event | None = None,
            log_msg: str | None = None,
    ):
        self.id = str(random.randint(0, 2 << 30)) if torch_event is not None else '-'
        self.__torch_event = torch_event
        self.__log_msg = log_msg

    def record(self):
        if self.__torch_event is not None:
            self.__torch_event.record()

    def wait(self, stream: torch.Stream, log_msg: str | None = None):
        if log_msg is None:
            log_msg = ""

        if self.__log_msg is not None:
            log_msg = f"{log_msg}, {self.__log_msg}"

        if self.__torch_event is not None and stream is not None:
            stream.wait_event(self.__torch_event)

        log_msg = f"{log_msg}, {self.id}"
        log(log_msg)

    def synchronize(self, log_msg: str | None = None):
        if log_msg is None:
            log_msg = ""

        if self.__log_msg is not None:
            log_msg = f"{log_msg}, {self.__log_msg}"

        if self.__torch_event is not None:
            if self.__torch_event.query():
                log_msg = f"{log_msg}, no-op"
            else:
                self.__torch_event.synchronize()
                log_msg = f"{log_msg}, syncing"
        else:
            log_msg = f"{log_msg}, skipping"

        log_msg = f"{log_msg}, {self.id}"
        log(log_msg)

    def __repr__(self) -> str:
        if self.__torch_event is None:
            return "event(None)"
        else:
            return f"event({self.__log_msg}, done={self.__torch_event.query()})"


class LayerOffloadStrategy:
    def __init__(
            self,
            layers: list[nn.Module],
            layer_offload_fraction: float,
    ):
        layer_bytes = [sum([get_offload_tensor_bytes(x) for x in layer.modules()]) for layer in layers]
        total_bytes = sum(layer_bytes)
        target_loaded_bytes = int(total_bytes * (1.0 - layer_offload_fraction))

        # calculate min number of loaded layers at the start
        self.initial_loaded_layers = self.__get_layers_below(
            layer_bytes=layer_bytes,
            start_layer=0,
            max_bytes=target_loaded_bytes,
            is_forward=True,
            is_cyclic=False,
        )

        # the offloading strategy has 3 cases:
        # case 1, forward pass, followed by a backward pass:
        #     do not offload the last layers, they will be needed immediately
        # case 2, forward pass,  followed by another forward pass:
        #     start loading the first layers when executing the last layers
        # case 3, backward pass:
        #     same as case 1, but in reversed order

        # calculate a list of loaded layers before execution of each layer
        self.forward_backward_loaded_layers = [self.__get_layers_below(
            layer_bytes=layer_bytes,
            start_layer=i,
            max_bytes=target_loaded_bytes,
            is_forward=True,
            is_cyclic=False,
        ) for i in range(len(layers))]

        self.forward_forward_loaded_layers = [self.__get_layers_below(
            layer_bytes=layer_bytes,
            start_layer=i,
            max_bytes=target_loaded_bytes,
            is_forward=True,
            is_cyclic=True,
        ) for i in range(len(layers))]

        self.backward_forward_loaded_layers = [self.__get_layers_below(
            layer_bytes=layer_bytes,
            start_layer=i,
            max_bytes=target_loaded_bytes,
            is_forward=False,
            is_cyclic=False,
        ) for i in range(len(layers))]

        all_loaded_layers = self.forward_backward_loaded_layers \
                            + self.forward_forward_loaded_layers \
                            + self.backward_forward_loaded_layers

        self.max_loaded_bytes = max(sum([layer_bytes[i] for i in loaded_layers]) for loaded_layers in all_loaded_layers)
        min_loaded_bytes = min(sum([layer_bytes[i] for i in loaded_layers]) for loaded_layers in all_loaded_layers)
        self.max_offloaded_bytes = total_bytes - min_loaded_bytes + max(layer_bytes)

    @staticmethod
    def __get_layers_below(
            layer_bytes: list[int],
            start_layer: int,
            max_bytes: int,
            is_forward: bool,
            is_cyclic: bool,
    ) -> list[int]:
        accumulator = 0
        layers = []
        if is_forward and is_cyclic:
            for i in range(start_layer, len(layer_bytes)):
                accumulator += layer_bytes[i]
                if accumulator > max_bytes and len(layers) >= 2:
                    break
                layers.append(i)
            for i in range(start_layer):
                accumulator += layer_bytes[i]
                if accumulator > max_bytes and len(layers) >= 2:
                    break
                layers.append(i)
        elif is_forward and not is_cyclic:
            for i in range(start_layer, len(layer_bytes)):
                accumulator += layer_bytes[i]
                if accumulator > max_bytes and len(layers) >= 2:
                    break
                layers.append(i)
            for i in range(start_layer - 1, -1, -1):
                accumulator += layer_bytes[i]
                if accumulator > max_bytes and len(layers) >= 2:
                    break
                layers.append(i)
        else:
            for i in range(start_layer, -1, -1):
                accumulator += layer_bytes[i]
                if accumulator > max_bytes and len(layers) >= 2:
                    break
                layers.append(i)
            for i in range(start_layer + 1, len(layer_bytes)):
                accumulator += layer_bytes[i]
                if accumulator > max_bytes and len(layers) >= 2:
                    break
                layers.append(i)
        return sorted(layers)

    def get_layers_to_offload(
            self,
            layer_index: int,
            is_forward: bool,
            is_next_forward: bool,
            loaded_layers: list[int],
    ) -> list[int]:
        layers = []
        if is_forward and is_next_forward:
            layers = sorted([i for i in loaded_layers if i not in self.forward_forward_loaded_layers[layer_index]])
        if is_forward and not is_next_forward:
            layers = sorted([i for i in loaded_layers if i not in self.forward_backward_loaded_layers[layer_index]])
        if not is_forward:
            layers = sorted([i for i in loaded_layers if i not in self.backward_forward_loaded_layers[layer_index]],
                            reverse=True)

        if is_forward:
            return [x for x in layers if x >= layer_index] + [x for x in layers if x < layer_index]
        else:
            return [x for x in layers if x < layer_index] + [x for x in layers if x >= layer_index]

    def get_layers_to_load(
            self,
            layer_index: int,
            is_forward: bool,
            is_next_forward: bool,
            loaded_layers: list[int],
    ) -> list[int]:
        layers = []
        if is_forward and is_next_forward:
            layers = sorted([i for i in self.forward_forward_loaded_layers[layer_index] if i not in loaded_layers])
        if is_forward and not is_next_forward:
            layers = sorted([i for i in self.forward_backward_loaded_layers[layer_index] if i not in loaded_layers])
        if not is_forward:
            layers = sorted([i for i in self.backward_forward_loaded_layers[layer_index] if i not in loaded_layers],
                            reverse=True)

        if is_forward:
            return [x for x in layers if x >= layer_index] + [x for x in layers if x < layer_index]
        else:
            return [x for x in layers if x < layer_index] + [x for x in layers if x >= layer_index]


class _BoundaryActivation:
    # Offloaded copy of a saved activation on the boundary path. Copy semantics (never mutate the saved
    # tensor in place - it may be shared across blocks, e.g. a conditioning embedding). cpu holds the
    # temp-device copy; gpu the reloaded train-device copy; event marks the reload transfer.
    def __init__(self):
        self.cpu = None
        self.gpu = None
        self.event = None
        # the source tensor's stride, restored on reload. A saved activation is often a permuted view
        # (an attention output, a head-split), and the compiled backward asserts on exact strides, so
        # handing it back contiguous fails assert_size_stride rather than silently computing wrong.
        self.stride = None
        # whether the source's storage region was dense, so both legs could move it as flat storage rather
        # than reordering elements through a copy kernel.
        self.dense = False


class LayerOffloadConductor:
    __module: nn.Module

    __layers: list[nn.Module]
    __layer_device_map: list[torch.device | None]
    __layer_offload_fraction: float


    __train_device: torch.device
    __temp_device: torch.device

    __offload_activations: bool
    __offload_layers: bool
    __async_transfer: bool

    __train_stream: torch.Stream | None
    __layer_transfer_stream: torch.Stream | None
    __activations_transfer_stream: torch.Stream | None

    __train_device_layer_allocator: StaticLayerAllocator
    __temp_device_layer_allocator: StaticLayerAllocator
    __temp_device_activations_allocator: StaticActivationAllocator

    __layer_train_event_map: list[SyncEvent]
    __layer_transfer_event_map: list[SyncEvent]


    __offload_strategy = LayerOffloadStrategy | None
    __is_forward_pass: bool
    __backward_follows: bool

    __is_active: bool

    __deferred_layers: list[int]

    __config: TrainConfig

    def __init__(
            self,
            module: nn.Module,
            config: TrainConfig,
            part: TrainModelPartConfig,
    ):
        super().__init__()

        self.__module = module

        self.__layers = []
        self.__layer_device_map = []
        self.__layer_offload_fraction = part.offload_fraction


        self.__train_device = torch.device(config.train_device)
        self.__temp_device = torch.device(config.temp_device)

        self.__offload_activations = part.activation_offloading
        self.__offload_layers = part.offload_fraction > 0
        self.__async_transfer = self.__train_device.type == "cuda" and config.async_offloading

        if self.__async_transfer:
            self.__train_stream = torch.cuda.default_stream(self.__train_device)
            self.__layer_transfer_stream = torch.cuda.Stream(self.__train_device)
            self.__activations_transfer_stream = torch.cuda.Stream(self.__train_device)
        else:
            self.__train_stream = None
            self.__layer_transfer_stream = None
            self.__activations_transfer_stream = None

        self.__train_device_layer_allocator = StaticLayerAllocator(self.__train_device)
        self.__temp_device_layer_allocator = StaticLayerAllocator(self.__temp_device)
        self.__temp_device_activations_allocator = StaticActivationAllocator(self.__temp_device)

        self.__layer_train_event_map = []
        self.__layer_transfer_event_map = []

        self.__boundary_activations = {}

        self.__offload_strategy = None
        self.__is_forward_pass = False
        self.__backward_follows = False
        self.__inflight_call_events = deque()
        self.__inflight_transfer_events = deque()
        self.__warned_non_dense = False

        self.__is_active = False

        self.__deferred_layers = []

        self.__config = config

    def offload_activated(self) -> bool:
        return self.__offload_activations or self.__offload_layers

    def offloads_activations(self) -> bool:
        return self.__offload_activations

    def to(self, device: torch.device):
        torch_gc()

        self.__wait_all_layer_transfers()

        if device_equals(device, self.__temp_device):
            log("to temp device")

            # deallocate the cache before to take advantage of the gc
            self.__train_device_layer_allocator.deallocate_cache()
            self.__temp_device_layer_allocator.deallocate_cache()
            self.__temp_device_activations_allocator.deallocate_cache()

            self.__module_to_device_except_layers(self.__temp_device)
            for layer_index, layer in enumerate(self.__layers):
                self.__layers[layer_index].to(self.__temp_device)
                for module in layer.modules():
                    offload_quantized(module, self.__temp_device, allocator=clone_tensor_allocator)
                self.__layer_device_map[layer_index] = None

            self.__is_active = False

        elif device_equals(device, self.__train_device):
            log("to train device")

            self.__offload_strategy = LayerOffloadStrategy(self.__layers, self.__layer_offload_fraction)

            self.__train_device_layer_allocator.allocate_cache(
                self.__layers, self.__offload_strategy.max_loaded_bytes)
            self.__temp_device_layer_allocator.allocate_cache(
                self.__layers, self.__offload_strategy.max_offloaded_bytes)
            self.__module_to_device_except_layers(self.__train_device)

            # move all layers to the train device, then move offloadable tensors back to the temp device
            for layer_index, layer in enumerate(self.__layers):
                if self.__layer_device_map[layer_index] is None:
                    log(f"layer {layer_index} to train device")
                    layer.to(self.__train_device)

                    if layer_index in self.__offload_strategy.initial_loaded_layers:
                        allocator = self.__train_device_layer_allocator.get_allocator(
                            layer_index, allocate_forward=True)
                        for module in layer.modules():
                            offload_quantized(module, self.__train_device, allocator=allocator.allocate_like)
                        self.__layer_device_map[layer_index] = self.__train_device
                    else:
                        allocator = self.__temp_device_layer_allocator.get_allocator(layer_index, allocate_forward=True)
                        for module in layer.modules():
                            offload_quantized(module, self.__temp_device, allocator=allocator.allocate_like)
                        self.__layer_device_map[layer_index] = self.__temp_device

                    if self.__async_transfer:
                        event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
                        self.__layer_train_event_map[layer_index] = event

            self.__is_active = True

        torch_gc()

    def add_layer(self, layer: nn.Module):
        self.__layers.append(layer)
        self.__layer_device_map.append(None)
        self.__layer_train_event_map.append(SyncEvent())
        self.__layer_transfer_event_map.append(SyncEvent())

    def start_forward(self, backward_follows: bool):
        log("starting forward")

        if not self.__is_active:
            return

        if self.__async_transfer:
            self.__layer_transfer_stream.wait_stream(self.__train_stream)
        self.__wait_all_layer_transfers()
        self.__clear_activations()

        self.__is_forward_pass = True
        self.__backward_follows = backward_follows
        # events from the previous step refer to work the GPU has long finished; carrying them over would
        # make the first calls of this pass wait on stale entries.
        self.__inflight_call_events.clear()
        self.__inflight_transfer_events.clear()

    def __schedule_layer_offload(self, layer_index: int, is_forward: bool, is_next_forward: bool):
        # windowed layer load/offload schedule around layer_index, in the direction the caller is
        # running: the boundaries know whether they are in the forward or the backward pass.
        if not self.__offload_layers:
            return

        self.__wait_layer_transfer(layer_index)

        self.__schedule_deferred_layers_to_temp(except_layer=layer_index)
        for i in self.__offload_strategy.get_layers_to_offload(
                layer_index=layer_index,
                is_forward=is_forward,
                is_next_forward=is_next_forward,
                loaded_layers=self.__get_loaded_layers(),
        ):
            self.__schedule_layer_to(i, self.__temp_device, is_forward=is_forward)

        for i in self.__offload_strategy.get_layers_to_load(
                layer_index=layer_index,
                is_forward=is_forward,
                is_next_forward=is_next_forward,
                loaded_layers=self.__get_loaded_layers(),
        ):
            self.__schedule_layer_to(i, self.__train_device, is_forward=is_forward)

    def before_layer(self, layer_index: int, is_forward: bool):
        # called before a block runs. Waits for this layer's transfer and slides the load window.
        if not self.__is_active:
            return

        # block until compute and activation transfers have caught up to within MAX_LAYER_CALLS_IN_FLIGHT,
        # so the floating activations stay bounded.
        if MAX_LAYER_CALLS_IN_FLIGHT > 0:
            queues = (("compute", self.__inflight_call_events), ("transfer", self.__inflight_transfer_events))
            for name, queue in queues:
                while len(queue) > MAX_LAYER_CALLS_IN_FLIGHT:
                    queue.popleft().synchronize(f"layer-calls-in-flight cap ({name})")

        self.__schedule_layer_offload(layer_index, is_forward, not self.__backward_follows)

    def after_layer(self, layer_index: int, activations: Any):
        # called once this layer's compute is enqueued. Keeps the block's output/grad alive until the train
        # stream reaches it, and records the event a later offload of this layer has to wait for.
        if not self.__is_active or not self.__async_transfer:
            return
        tensors_record_stream(self.__train_stream, activations)
        event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
        self.__layer_train_event_map[layer_index] = event
        # the same event, queued in call order, is what the run-ahead cap waits on
        self.__inflight_call_events.append(event)

        # Second marker, on the activations transfer stream, which trails the train stream by its own queue
        # (measured ~106ms median, ~12 layer-calls of float against a cap of 2). An offloaded activation's
        # source block is pinned until its D2H actually executes, so this is the distance that holds memory.
        if self.__offload_activations:
            self.__inflight_transfer_events.append(
                SyncEvent(self.__activations_transfer_stream.record_event(), "activations transfer"))

    def __get_loaded_layers(self) -> list[int]:
        return [i for i in range(len(self.__layers)) if device_equals(self.__layer_device_map[i], self.__train_device)]

    def __module_to_device_except_layers(
            self,
            device: torch.device,
    ):
        sub_module_parameters = set(sum([list(x.parameters()) for x in self.__layers], []))

        def convert(t):
            if t in sub_module_parameters or t.is_meta:
                return t

            return t.to(device=device)

        self.__module._apply(convert)

    def __clear_activations(self):
        self.__boundary_activations.clear()
        self.__temp_device_activations_allocator.deallocate()

    def __wait_all_layer_train(self):
        for layer_index in range(len(self.__layers)):
            self.__wait_layer_train(layer_index)

    def __wait_all_layer_transfers(self):
        for layer_index in range(len(self.__layers)):
            self.__wait_layer_transfer(layer_index)

    def __wait_layer_train(self, layer_index: int):
        self.__layer_train_event_map[layer_index] \
            .wait(self.__layer_transfer_stream, f"wait layer train {layer_index}")
        self.__layer_train_event_map[layer_index] = SyncEvent()

    def __wait_layer_transfer(self, layer_index: int):
        if self.__async_transfer:
            self.__layer_transfer_event_map[layer_index] \
                .wait(self.__train_stream, f"wait layer transfer {layer_index}")
            self.__layer_transfer_event_map[layer_index] = SyncEvent()

    def __schedule_layer_to(
            self,
            layer_index: int,
            device: torch.device,
            is_forward: bool,
    ):
        current_device = self.__layer_device_map[layer_index]
        if device_equals(device, current_device):
            log(f"schedule layer {layer_index} to {str(device)}, skipping")
            return

        layer_deallocator = self.__temp_device_layer_allocator \
            if device_equals(device, self.__train_device) \
            else self.__train_device_layer_allocator

        layer_allocator = self.__train_device_layer_allocator \
            if device_equals(device, self.__train_device) \
            else self.__temp_device_layer_allocator
        allocator = layer_allocator.get_allocator(layer_index, is_forward)

        allocator_fn = allocator.allocate_like if allocator is not None else None

        if not is_forward and device_equals(device, self.__temp_device):
            layer = self.__layers[layer_index]
            for module in layer.modules():
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        #Layers to be offloaded usually do not have gradients. Model weights only have gradients in full-finetuning,
                        #and then a fused backpass is required for offloading, which sets all gradients to None before a layer is offloaded.
                        #There is only once exception:
                        #In Multi-GPU training, when the gradient reduction has been started during the fused back pass, but
                        #has not finished yet. The gradients are then set to None during the backward of one of the next layers.
                        #Record which layers were ready to be offloaded, and offload them later:
                        if (not self.__config.multi_gpu or not self.__config.optimizer.fused_back_pass
                            or not self.__config.fused_gradient_reduce or not self.__config.async_gradient_reduce):
                            raise RuntimeError("Gradients are still active while attempting to offload a layer")

                        #TODO deferring layer offloading appears to work for multi-GPU training, but there might be edge cases because offloading depends on the exact same order of layer execution. It's possible that when communication between GPU lags, more layers are deferred and this fails silently.
                        self.__deferred_layers.append(layer_index)
                        return

        with create_stream_context(self.__layer_transfer_stream):
            self.__wait_layer_train(layer_index)
            layer = self.__layers[layer_index]
            for module in layer.modules():
                offload_quantized(module, device, non_blocking=self.__async_transfer, allocator=allocator_fn)

            layer_deallocator.deallocate_layer(layer_index, deallocate_forward=is_forward)

            if self.__async_transfer:
                event = SyncEvent(self.__layer_transfer_stream.record_event(), f"transfer to {device}")
                self.__layer_transfer_event_map[layer_index] = event
                log(f"schedule layer {layer_index} to {str(device)}, {event}")
            else:
                log(f"schedule layer {layer_index} to {str(device)}")

            self.__layer_device_map[layer_index] = device

    def __schedule_deferred_layers_to_temp(
            self,
            except_layer: int,
    ):
        layers = self.__deferred_layers
        self.__deferred_layers = []
        for layer_index in layers:
            if layer_index == except_layer:
                #don't offload this layer, because it is needed now #TODO can this even happen?
                continue
            self.__schedule_layer_to(layer_index, device=self.__temp_device, is_forward=False)

    def pack_activation(self, layer_index: int, tensor: torch.Tensor):
        # Copies rather than moving in place, so a tensor still referenced by other blocks (e.g. a shared
        # conditioning embedding) is never corrupted. The caller selects which tensors to offload.
        if not self.__is_active or not self.__offload_activations \
                or not device_equals(tensor.device, self.__train_device) \
                or layer_index >= len(self.__layers) - MAX_LAYER_CALLS_IN_FLIGHT:
            return tensor

        handle = _BoundaryActivation()
        handle.stride = tensor.stride()
        # Transfer the storage as-is when the tensor is dense, so both sides of the copy are contiguous and
        # equal-length and the transfer stays a DMA. Non-dense tensors have no flat view and fall back to the
        # logical copy, which reorders elements through a kernel but is always correct.
        source = flat_storage_view(tensor)
        handle.dense = source is not None
        if source is None:
            source = tensor
            if not self.__warned_non_dense:
                # not expected with the current save set (SDPA and mm outputs are fresh allocations), but a
                # chunk or slice would land here, so say so rather than silently paying the kernel path.
                # Warn rather than raise: a save set that widens to include a view should get slower, not
                # abort a training run.
                self.__warned_non_dense = True
                print("non-dense activation offloaded via the logical copy path: "
                      f"layer {layer_index}, shape {tuple(tensor.shape)}, stride {tensor.stride()}")

        with create_stream_context(self.__activations_transfer_stream):
            if self.__async_transfer:
                self.__activations_transfer_stream.wait_stream(self.__train_stream)
            self.__temp_device_activations_allocator.reserve_cache([tensor])
            handle.cpu = self.__temp_device_activations_allocator.allocate_like(tensor)
            destination = handle.cpu.view(-1) if handle.dense else handle.cpu
            destination.copy_(source, non_blocking=self.__async_transfer)
            if self.__async_transfer:
                tensors_record_stream(self.__activations_transfer_stream, tensor)  # source alive until copied

        self.__boundary_activations.setdefault(layer_index, []).append(handle)
        return handle

    def __reload_activation(self, handle: '_BoundaryActivation'):
        if handle.gpu is not None:
            return
        # Allocate under the train stream: the allocator's free lists are per-stream, so a destination
        # homed on the transfer stream can never be reused by compute and forms a second segment pool.
        # Dense tensors allocate contiguous and get the recorded layout as_strided over them, so the DMA
        # moves storage without reordering. empty_strided is the non-dense fallback.
        with create_stream_context(self.__train_stream):
            if handle.dense:
                flat = torch.empty(handle.cpu.numel(), dtype=handle.cpu.dtype, device=self.__train_device)
                handle.gpu = torch.as_strided(flat, handle.cpu.shape, handle.stride)
            else:
                handle.gpu = torch.empty_strided(
                    handle.cpu.shape, handle.stride, dtype=handle.cpu.dtype, device=self.__train_device)

        # the allocator may have just reclaimed this block from still-queued train work, which the H2D on
        # the transfer stream would otherwise overwrite. Recorded, not waited on here, so the transfer
        # still overlaps the current layer's compute.
        event = SyncEvent(self.__train_stream.record_event(), "train before activation reload") \
            if self.__async_transfer else None

        with create_stream_context(self.__activations_transfer_stream):
            if event is not None:
                event.wait(self.__activations_transfer_stream)
            if handle.dense:
                flat_storage_view(handle.gpu).copy_(handle.cpu.view(-1), non_blocking=self.__async_transfer)
            else:
                handle.gpu.copy_(handle.cpu, non_blocking=self.__async_transfer)
            if self.__async_transfer:
                tensors_record_stream(self.__activations_transfer_stream, handle.gpu)
                handle.event = SyncEvent(self.__activations_transfer_stream.record_event())

    def prefetch_activations(self, layer_index: int):
        # reload a block's offloaded activations one block ahead so unpack only waits on the transfer
        if not self.__is_active or not self.__offload_activations:
            return
        for handle in self.__boundary_activations.get(layer_index, []):
            self.__reload_activation(handle)

    def unpack_activation(self, handle: Any):
        if not isinstance(handle, _BoundaryActivation):
            return handle  # was not offloaded
        self.__reload_activation(handle)  # no-op if already prefetched
        if self.__async_transfer and handle.event is not None:
            handle.event.wait(self.__train_stream)
            tensors_record_stream(self.__train_stream, handle.gpu)
        gpu = handle.gpu
        handle.gpu = None
        handle.event = None
        return gpu
