import math
import random
from collections.abc import Callable
from typing import Any

from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.disk_stream import _is_evicted, evict_to_meta
from modules.util.enum.DataType import DataType
from modules.util.quantization_util import (
    get_offload_tensor_bytes,
    get_offload_tensors,
    is_quantized_module,
    offload_quantized,
)
from modules.util.torch_util import (
    create_mem_pool,
    create_stream_context,
    device_equals,
    get_tensor_data,
    mem_pool_context,
    pin_tensor_,
    replace_tensors_,
    tensors_match_device,
    tensors_record_stream,
    tensors_to_device_,
    torch_gc,
    unpin_tensor_,
)

import torch
from torch import nn

from tqdm import tqdm

MESSAGES = []


def log(msg: str = ''):
    pass
    # print(msg)
    # MESSAGES.append(msg)


def clone_tensor_allocator(tensor: torch.Tensor) -> torch.Tensor:
    # clones a tensor into a new memory location to remove all memory dependencies between tensors
    return tensor.clone()


# allocate_like places each cached tensor at an aligned offset, wasting up to this many bytes per tensor.
# also the reserved size at the start of each cache tensor (see allocate_like); must stay >= 2 so no view
# ever lands at storage_offset 0 or 1, the two values torch.compile bakes into separate specialized graphs
TENSOR_ALIGNMENT_BYTES = 16


def align_up(number: int) -> int:
    return number + (TENSOR_ALIGNMENT_BYTES - (number % TENSOR_ALIGNMENT_BYTES)) % TENSOR_ALIGNMENT_BYTES


def align_down(number: int) -> int:
    return number - (number % TENSOR_ALIGNMENT_BYTES)


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
            # never hand out views at storage_offset 0: torch.compile creates a 0/1-specialized
            # symbol for the storage_offset of any tensor with a dynamic dim, so an offset-0 view
            # needs its own graph while one "2 <= offset" guard covers all
            # other placements. keeping every view past the first alignment slot avoids those
            # recompiles, and costs each tensor at most its alignment budget (the first tensor in a
            # cache tensor previously wasted 0 of it)
            cache_tensor_allocation_end = max(align_up(self.__allocation_end % cache_tensor_size), TENSOR_ALIGNMENT_BYTES)

            if cache_tensor_allocation_end + num_bytes > cache_tensor_size:
                # move to the start of the next cache tensor
                cache_tensor_index += 1
                cache_tensor_allocation_end = TENSOR_ALIGNMENT_BYTES
            if cache_tensor_index * cache_tensor_size + cache_tensor_allocation_end + num_bytes > total_cache_bytes:
                # move to the first cache tensor
                cache_tensor_index = 0
                cache_tensor_allocation_end = TENSOR_ALIGNMENT_BYTES

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

            # "< TENSOR_ALIGNMENT_BYTES" instead of "< 0": the first alignment slot of every cache
            # tensor is reserved so no view lands at storage_offset 0 (see the forward-direction
            # comment above)
            if cache_tensor_allocation_start - num_bytes < TENSOR_ALIGNMENT_BYTES:
                # move to the end of the previous cache tensor
                cache_tensor_index -= 1
                cache_tensor_allocation_start = cache_tensor_size
            if cache_tensor_index < 0:
                # move to the first cache tensor
                cache_tensor_index = len(self.__layer_allocator.cache_tensors) - 1
                cache_tensor_allocation_start = cache_tensor_size

            new_allocation_start = align_down(cache_tensor_allocation_start - num_bytes)
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

        self.__mem_pool = None

    def allocate_cache(self, layers: list[nn.Module], target_bytes: int, streaming: bool, cache_in_ram: bool):
        if not self.__allocate_statically or any(x is not None for x in self.cache_tensors):
            return

        log(f"allocating cache on device {self.device}")

        # keep the cache tensor in its own MemPool to avoid fragmenting the next cycle's allocation
        if self.__mem_pool is None:
            self.__mem_pool = create_mem_pool(self.device)

        self.__max_tensor_bytes = 0
        self.__layer_bytes = []
        total_tensors = 0  # count of individual offload tensors == number of allocate_like calls == alignment slots
        for layer in layers:
            layer_tensor_bytes = [get_offload_tensor_bytes(x) for x in layer.modules()]
            total_tensors += sum(len(get_offload_tensors(x)) for x in layer.modules())
            self.__max_tensor_bytes = max(self.__max_tensor_bytes, *layer_tensor_bytes)
            self.__layer_bytes.append(sum(layer_tensor_bytes))

        cache_bytes = target_bytes
        if self.device.type == "cuda":
            # single cache tensor on the GPU: a large cuda allocation is page-mapped (assembled from scattered
            # physical pages), so one buffer allocates as readily as many and packs with no inter-chunk tail waste.
            # The GPU cache is filled one layer at a time from the CPU, so the destination buffer and a full
            # resident source never coexist on the device -- no peak-doubling to guard against here.
            num_cache_tensors = 1
        elif streaming and not cache_in_ram:
            # host/pinned cache, disk-streaming with cache_in_ram off: layers stream+quantize straight from the
            # checkpoint and evict back to meta, so no resident copy ever coexists with the pinned cache -- none of the
            # peak-doubling that justifies chunking below. A single large pinned buffer is fine: pin_tensor_ page-locks
            # the existing scattered pages in place, and the CPU allocator has no pool to fragment. Same as the GPU cache.
            num_cache_tensors = 1
        else:
            # host/pinned cache, resident model (classic offload, or streaming with cache_in_ram on): the chunks are
            # allocated lazily (per ensure_allocation) to cap peak host RAM while the resident model is copied into
            # the pinned cache (and, on evict, cloned back out of it), which a single eager buffer would roughly double.
            num_cache_tensors = min(
                # no more than 10% overhead
                math.ceil(int(cache_bytes * 0.10) / self.__max_tensor_bytes),
                # at least twice self.__max_tensor_bytes for each tensor
                math.ceil(cache_bytes / (self.__max_tensor_bytes * 2)),
                # no more than 10 cache tensors
                10
            )
        # the alignment budget must cover EVERY tensor packed into a cache tensor: allocate_like wastes up to
        # TENSOR_ALIGNMENT_BYTES per tensor and the ring wrap is unguarded, so a fixed total would silently
        # overwrite live weights once a cache tensor holds enough tensors. Size it from the actual tensor count.
        alignment_bytes = TENSOR_ALIGNMENT_BYTES * total_tensors
        # add self.__max_tensor_bytes so even the largest tensor fits in the space left after a ring wrap
        self.cache_tensor_size = math.ceil(cache_bytes / num_cache_tensors) + self.__max_tensor_bytes + alignment_bytes

        self.__tensor_allocators = [None] * len(layers)
        self.cache_tensors = [None] * num_cache_tensors
        self.allocation_start = 0
        self.allocation_end = 0

    def ensure_allocation(self, cache_tensor_index: int):
        if self.cache_tensors[cache_tensor_index] is None:
            torch_gc()

            # create the cache tensor inside the MemPool so it lands in the pool's isolated segments. the buffers
            # are allocated lazily here (allocate_cache only sizes them), so the pool context wraps this
            # allocation rather than allocate_cache.
            with mem_pool_context(self.__mem_pool):
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
        # the loop above leaves `cache_tensor` bound to the last tensor; clear it so that stray reference can't
        # keep the MemPool alive through the torch_gc below
        cache_tensor = None

        # drop the MemPool once its tensors are freed so its now-empty segments return to the driver for the
        # default pool; a fresh one is created on the next allocate_cache.
        if self.__mem_pool is not None:
            self.__mem_pool = None
            torch_gc()

    @property
    def mem_pool(self):
        # the MemPool holding this allocator's cache tensor(s); also used to keep the conductor's resident non-layer
        # remainder out of the default pool. allocate_cache creates it before the materialize layer loop; create it
        # here too in case a caller reaches for it first. deallocate_cache drops it (static allocators only).
        if self.__mem_pool is None:
            self.__mem_pool = create_mem_pool(self.device)
        return self.__mem_pool

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

    def reserve_cache(self, tensors: list[torch.Tensor]):
        num_bytes = sum(tensor.element_size() * tensor.numel() for tensor in tensors) \
                    + len(tensors) * TENSOR_ALIGNMENT_BYTES  # add enough padding for alignment

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
        self.__current_cache_tensor_offset += align_up(num_bytes)

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


class LayerOffloadConductor:
    __module: nn.Module

    __layers: list[nn.Module]
    __layer_device_map: list[torch.device | None]
    __layer_offload_fraction: float

    __layer_activations_included_offload_param_indices_map: list[list[int]]

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

    __activations_map: dict[int, Any]
    __call_index_layer_index_map: dict[int, int]
    __activations_transfer_event_map: dict[int, SyncEvent]

    __offload_strategy = LayerOffloadStrategy | None
    __is_forward_pass: bool
    __keep_graph: bool

    __materialized: bool

    __deferred_layers: list[int]

    __config: TrainConfig

    __disk_remainder_materialized: bool         # whether the non-layer remainder (embedders/norms/proj) has been streamed since the last evict
    __disk_layer_key_prefixes: list[str]        # per-layer (indexed like __layers) checkpoint-absolute path, so a single layer subtree can be streamed on its own
    __disk_module_name_by_id: dict[int, str]    # module-name snapshot taken pre-wrapping, used to build the key prefixes above

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

        self.__layer_activations_included_offload_param_indices_map = []

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

        self.__activations_map = {}
        self.__call_index_layer_index_map = {}
        self.__activations_transfer_event_map = {}

        self.__offload_strategy = None
        self.__is_forward_pass = False
        self.__keep_graph = False

        self.__materialized = False

        self.__deferred_layers = []

        self.__config = config

        self.__disk_remainder_materialized = False
        self.__disk_layer_key_prefixes = []
        self.__disk_module_name_by_id = {id(m): name for name, m in module.named_modules()}

    def offload_activated(self) -> bool:
        return self.__offload_activations or self.__offload_layers

    def evict(self, to_meta: bool = False):
        torch_gc()

        self.__wait_all_layer_transfers()
        self.__wait_all_activation_transfers()

        log("to temp device")

        if to_meta:
            if self.__materialized:
                self.__evict_to_meta()
        else:
            self.__evict_to_temp()

    def __evict_to_temp(self):
        # move every layer and the non-layer remainder back to the temp device and free the static caches (the
        # non-disk eviction path). Also the rollback for a resident conductor whose materialize() raised partway.
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

        self.__materialized = False

    def materialize(
            self, train_dtype: DataType | None = None, name: str | None = None,
            materialize_fn: Callable | None = None, cache_in_ram: bool = True):
        torch_gc()

        self.__wait_all_layer_transfers()
        self.__wait_all_activation_transfers()

        streaming = materialize_fn is not None

        log("to train device")

        try:
            self.__offload_strategy = LayerOffloadStrategy(self.__layers, self.__layer_offload_fraction)

            self.__train_device_layer_allocator.allocate_cache(
                self.__layers, self.__offload_strategy.max_loaded_bytes, streaming=streaming, cache_in_ram=cache_in_ram)
            self.__temp_device_layer_allocator.allocate_cache(
                self.__layers, self.__offload_strategy.max_offloaded_bytes, streaming=streaming, cache_in_ram=cache_in_ram)
            # place the resident non-layer remainder onto the train device. When streaming, route it into the conductor
            # pool: on a warm cache_in_ram re-activate it comes from cpu/temp and lands there directly (no default-pool
            # copy to relocate); on a cold stream it is still meta here and gets skipped, then streamed below.
            self.__module_to_device_except_layers(
                self.__train_device,
                pool=self.__train_device_layer_allocator.mem_pool if streaming else None)

            cold_layers = sum(1 for i, layer in enumerate(self.__layers)
                              if self.__layer_device_map[i] is None and _is_evicted(layer)) if streaming else 0
            disk_bar = tqdm(total=cold_layers, unit="layer", desc=f"streaming {name}", leave=False) \
                if cold_layers > 0 else None

            # bring each layer to the train device (streaming it if cold), then place it in its cache slot -- the GPU
            # cache for an initially-loaded layer, the temp-device cache for an offloaded one
            for layer_index, layer in enumerate(self.__layers):
                if self.__layer_device_map[layer_index] is None:
                    log(f"layer {layer_index} to train device")
                    if streaming and _is_evicted(layer):
                        # cold materialize from the checkpoint: stream+quantize this layer onto the train device
                        # (per tensor, quantized inline). The offload_quantized below then copies it into its
                        # static cache slot -- GPU cache for a loaded layer, pinned CPU cache for an offloaded one.
                        materialize_fn(
                            layer, self.__train_device, train_dtype,
                            self.__disk_layer_key_prefixes[layer_index])
                        if disk_bar is not None:
                            disk_bar.update(1)
                    else:
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

            if disk_bar is not None:
                disk_bar.close()

            if streaming and not self.__disk_remainder_materialized:
                # the non-layer remainder (embedders/norms/proj) is still meta the first time; stream it to the train
                # device now, where it stays resident. dest_pool routes the non-quantized weights straight into the
                # conductor pool so no model weight sits in the default pool (which the optimizer state and quantize
                # transients draw from). Quantized remainder weights pack in the default pool -- their dequant scratch
                # stays out of the pool -- and are relocated into it just below, once small.
                materialize_fn(self.__module, self.__train_device, train_dtype, "",
                               dest_pool=self.__train_device_layer_allocator.mem_pool)
                self.__disk_remainder_materialized = True
                self.__relocate_quantized_remainder_to_pool()
        except Exception:
            # a materialize that fails partway (typically OOM) leaves layers/cache tensors resident while
            # __materialized is still False, so a later evict() would skip them and strand that VRAM. Force the unit
            # back to its pre-materialize state, keyed on the actual weight state: a parameter still on meta means a
            # cold disk-stream was in flight, so meta is the only valid target (re-stream next time, lossless since
            # frozen); otherwise roll back to the temp device and keep the resident quantized copy.
            if any(parameter.is_meta for parameter in self.__module.parameters()):
                self.__evict_to_meta()
            else:
                self.__evict_to_temp()
            # the rollback helpers no longer gc, and no caller gc's a failed materialize -- reclaim the stranded VRAM
            # here before re-raising (evict() instead relies on BaseModel.evict()'s trailing gc).
            torch_gc()
            raise

        self.__materialized = True

    def add_layer(self, layer: nn.Module, included_offload_param_indices: list[int] = None):
        if included_offload_param_indices is None:
            included_offload_param_indices = []

        self.__layers.append(layer)
        self.__layer_device_map.append(None)
        self.__layer_train_event_map.append(SyncEvent())
        self.__layer_transfer_event_map.append(SyncEvent())
        # checkpoint-absolute path of this layer, for the per-layer disk stream (empty for a layer built outside self.__module)
        self.__disk_layer_key_prefixes.append(self.__disk_module_name_by_id.get(id(layer), ""))

        self.__layer_activations_included_offload_param_indices_map.append(included_offload_param_indices)

    def start_forward(self, keep_graph: bool):
        log("starting forward")

        if not self.__materialized:
            return

        if self.__async_transfer:
            self.__layer_transfer_stream.wait_stream(self.__train_stream)
        self.__wait_all_layer_transfers()
        self.__clear_activations()

        self.__is_forward_pass = True
        self.__keep_graph = keep_graph

    def before_layer(self, layer_index: int, call_index: int, activations: Any) -> Any:
        log()
        log(f"before layer {layer_index}, {call_index}")

        if not self.__materialized:
            return activations

        self.__call_index_layer_index_map[call_index] = layer_index

        if torch.is_grad_enabled() and self.__is_forward_pass:
            # Offloading can only be used with the use_reentrant=True checkpointing variant.
            # Gradients are only enabled during the back pass.
            log("starting backward")
            self.__is_forward_pass = False

        if self.__offload_activations and not self.__is_forward_pass:
            self.__wait_activations_transfer(call_index)

            tensor_indices = self.__layer_activations_included_offload_param_indices_map[layer_index]

            if call_index in self.__activations_map:
                # during the back pass, replace activations with saved acitvations
                replace_tensors_(activations, self.__activations_map.pop(call_index), tensor_indices)

            # if current activations are not on train_device, move them now
            if not tensors_match_device(
                    activations, self.__train_device,
                    tensor_indices):
                log(f"activations for layer {layer_index} not loaded to train device, transferring now")
                self.__schedule_activations_to_device(
                    activations, self.__train_device, call_index, wait_train_stream=False)
                self.__wait_activations_transfer(call_index)

            # schedule previous activations to the train device
            if call_index - 1 in self.__activations_map:
                self.__schedule_activations_to_device(
                    self.__activations_map[call_index - 1], self.__train_device, call_index - 1,
                    wait_train_stream=False)

        # schedule loading of the next layer and offloading of the previous layer
        if self.__offload_layers:
            self.__wait_layer_transfer(layer_index)

            self.__schedule_deferred_layers_to_temp(except_layer=layer_index)
            for i in self.__offload_strategy.get_layers_to_offload(
                    layer_index=layer_index,
                    is_forward=self.__is_forward_pass,
                    is_next_forward=not self.__keep_graph,
                    loaded_layers=self.__get_loaded_layers(),
            ):
                self.__schedule_layer_to(i, self.__temp_device, is_forward=self.__is_forward_pass)

            for i in self.__offload_strategy.get_layers_to_load(
                    layer_index=layer_index,
                    is_forward=self.__is_forward_pass,
                    is_next_forward=not self.__keep_graph,
                    loaded_layers=self.__get_loaded_layers(),
            ):
                self.__schedule_layer_to(i, self.__train_device, is_forward=self.__is_forward_pass)

        return activations

    def after_layer(self, layer_index: int, call_index: int, activations: Any):
        log(f"after layer {layer_index}, {call_index}")

        if not self.__materialized:
            return

        # record stream
        if self.__async_transfer:
            tensors_record_stream(self.__train_stream, activations)

        # save activations during the forward pass to make them accessible during the backward pass
        if self.__offload_activations and self.__keep_graph and self.__is_forward_pass:
            log(f"saving layer {call_index} activations for back pass")
            self.__activations_map[call_index] = activations
            self.__schedule_activations_to_device(activations, self.__temp_device, call_index, wait_train_stream=True)

        if self.__async_transfer:
            event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
            self.__layer_train_event_map[layer_index] = event

    def __get_loaded_layers(self) -> list[int]:
        return [i for i in range(len(self.__layers)) if device_equals(self.__layer_device_map[i], self.__train_device)]

    def __evict_to_meta(self):
        evict_to_meta(self.__module)
        for layer_index in range(len(self.__layers)):
            self.__layer_device_map[layer_index] = None
        self.__disk_remainder_materialized = False
        self.__train_device_layer_allocator.deallocate_cache()
        self.__temp_device_layer_allocator.deallocate_cache()
        self.__temp_device_activations_allocator.deallocate_cache()
        self.__materialized = False

    def __module_to_device_except_layers(
            self,
            device: torch.device,
            pool=None,
    ):
        sub_module_parameters = set(sum([list(x.parameters()) for x in self.__layers], []))

        def convert(t):
            if t in sub_module_parameters or t.is_meta:
                return t

            if pool is not None:
                # place the (already-final) non-layer remainder weight straight into the conductor's pool instead of
                # the default pool, which the optimizer state and quantize transients allocate from -- a weight left
                # there fragments it and strands the region when the remainder is evicted. A weight from cpu/temp (warm
                # cache_in_ram re-activate) lands in the pool directly; one already on the train device is relocated
                # with a clone.
                with mem_pool_context(pool):
                    return t.clone() if device_equals(t.device, device) else t.to(device=device)

            return t.to(device=device)

        self.__module._apply(convert)

    def __relocate_quantized_remainder_to_pool(self):
        # the cold remainder stream packs quantized non-layer weights (e.g. a tied lm_head) in the default pool so
        # their dequant scratch never enters the conductor pool. Copy just the packed weights into the pool now, so
        # no model weight is left in the default pool (where the optimizer state and quantize transients would
        # fragment/strand it). Small: the packed weights are a fraction of their fp size. Non-quantized remainder
        # weights were streamed straight into the pool (dest_pool) and are not touched here. Layer modules are
        # excluded -- they own their static cache slots -- matching __module_to_device_except_layers' scope.
        pool = self.__train_device_layer_allocator.mem_pool

        def pool_clone(tensor):
            with mem_pool_context(pool):
                return tensor.clone()

        layer_modules = {module for layer in self.__layers for module in layer.modules()}
        for module in self.__module.modules():
            if module not in layer_modules and is_quantized_module(module):
                offload_quantized(module, self.__train_device, allocator=pool_clone)

    def __clear_activations(self):
        self.__activations_map.clear()
        self.__call_index_layer_index_map.clear()
        self.__activations_transfer_event_map.clear()
        self.__temp_device_activations_allocator.deallocate()

    def __wait_all_layer_train(self):
        for layer_index in range(len(self.__layers)):
            self.__wait_layer_train(layer_index)

    def __wait_all_layer_transfers(self):
        for layer_index in range(len(self.__layers)):
            self.__wait_layer_transfer(layer_index)

    def __wait_all_activation_transfers(self):
        call_indices = list(self.__activations_transfer_event_map.keys())
        for call_index in call_indices:
            self.__wait_activations_transfer(call_index)

    def __wait_layer_train(self, layer_index: int):
        self.__layer_train_event_map[layer_index] \
            .wait(self.__layer_transfer_stream, f"wait layer train {layer_index}")
        self.__layer_train_event_map[layer_index] = SyncEvent()

    def __wait_layer_transfer(self, layer_index: int):
        if self.__async_transfer:
            self.__layer_transfer_event_map[layer_index] \
                .wait(self.__train_stream, f"wait layer transfer {layer_index}")
            self.__layer_transfer_event_map[layer_index] = SyncEvent()

    def __wait_activations_transfer(self, call_index: int):
        event = self.__activations_transfer_event_map.pop(call_index, None)

        if event is not None:
            event.wait(self.__train_stream, f"wait activations transfer {call_index}")

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

    def __schedule_activations_to_device(
            self,
            activations: Any,
            device: torch.device,
            call_index: int,
            wait_train_stream: bool,
    ):
        log(f"schedule {call_index} activations to {str(device)}")
        layer_index = self.__call_index_layer_index_map[call_index]

        activations_allocator = self.__temp_device_activations_allocator \
            if device_equals(device, self.__temp_device) \
            else None

        allocator_fn = activations_allocator.allocate_like if activations_allocator is not None else None

        event = None
        if wait_train_stream and self.__async_transfer:
            event = SyncEvent(self.__train_stream.record_event(), f"train before activations transfer {call_index}")

        with create_stream_context(self.__activations_transfer_stream):
            tensor_indices = self.__layer_activations_included_offload_param_indices_map[layer_index]

            if event is not None:
                event.wait(self.__activations_transfer_stream)

            tensors = get_tensor_data(activations, tensor_indices)
            if activations_allocator is not None:
                activations_allocator.reserve_cache(tensors)
            tensors_to_device_(activations, device, tensor_indices, non_blocking=self.__async_transfer, allocator=allocator_fn)

            if self.__async_transfer:
                tensors_record_stream(self.__activations_transfer_stream, tensors)
                self.__activations_transfer_event_map[call_index] = \
                    SyncEvent(self.__activations_transfer_stream.record_event(), f"transfer to {device}")

            del tensors
