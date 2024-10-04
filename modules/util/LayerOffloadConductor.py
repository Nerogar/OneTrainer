import random
from typing import Any

import torch
from torch import nn

from modules.util.config.TrainConfig import TrainConfig
from modules.util.quantization_util import get_offload_tensors, offload_quantized, get_offload_tensor_bytes
from modules.util.torch_util import (
    create_stream_context,
    device_equals,
    replace_tensors_,
    tensors_match_device,
    tensors_record_stream,
    tensors_to_device_,
    torch_gc,
    pin_tensor_, unpin_tensor_,
)

MESSAGES = []


def log(msg: str = ''):
    print(msg)
    # MESSAGES.append(msg)


class StaticTensorAllocator:
    def __init__(self, device: torch.device, tensor: torch.Tensor, just_allocated: bool):
        self.__device = device
        self.__tensor = tensor
        self.just_allocated = just_allocated

        self.__offset = 0

    def allocate_like(self, source_tensor: torch.Tensor) -> torch.Tensor:
        num_bytes = source_tensor.numel() * source_tensor.element_size()
        allocated = self.__tensor[self.__offset:self.__offset + num_bytes]
        self.__offset += num_bytes
        return allocated.view(dtype=source_tensor.dtype).view(size=source_tensor.shape)


class StaticLayerTensorAllocator:
    __device: torch.device
    __is_pinned: bool

    __cache_tensors: list[torch.Tensor | None] | None
    __cache_tensors_layer_index: list[torch.Tensor | None] | None
    __current_tensor: int
    __num_layers: int
    __max_layer_bytes: int

    def __init__(
            self,
            device: torch.device,
    ):
        self.__device = device
        self.__allocate_statically = True
        self.__is_pinned = device.type == "cpu"

        self.__cache_tensors = None
        self.__cache_tensors_layer_index = None
        self.__current_tensor = -1
        self.__num_layers = 0
        self.__max_layer_bytes = 0

    def allocate_cache(self, layers: list[nn.Module], num_layers: int):
        if not self.__allocate_statically or self.__cache_tensors is not None:
            return

        self.__num_layers = num_layers

        # This assumes that most layers are close in size.
        # Find a better allocation strategy once there are models with different architectures
        self.__max_layer_bytes = 0
        for layer in layers:
            self.__max_layer_bytes = \
                max(self.__max_layer_bytes, sum([get_offload_tensor_bytes(x) for x in layer.modules()], 0))

        self.__cache_tensors = [None] * self.__num_layers
        self.__cache_tensors_layer_index = [None] * self.__num_layers

        self.__current_tensor = -1

    def deallocate_cache(self, layer_index: int | None = None):
        if not self.__allocate_statically or self.__cache_tensors is None:
            return

        if layer_index is None:
            for tensor in self.__cache_tensors:
                if tensor is not None and self.__is_pinned:
                    unpin_tensor_(tensor)
            self.__cache_tensors = None
            self.__cache_tensors_layer_index = None
        else:
            for tensor_index in range(len(self.__cache_tensors)):
                if self.__cache_tensors_layer_index[tensor_index] == layer_index:
                    if self.__is_pinned:
                        unpin_tensor_(self.__cache_tensors[tensor_index])
                    self.__cache_tensors[tensor_index] = None
                    self.__cache_tensors_layer_index[tensor_index] = None

    def get_allocator(self, layer_index: int) -> StaticTensorAllocator | None:
        if self.__allocate_statically:
            # Always allocates the next tensor. The cache holds exactly the right amount of tensors,
            # which means that the next tensor is always free
            self.__current_tensor = (self.__current_tensor + 1) % len(self.__cache_tensors)
            self.__cache_tensors_layer_index[self.__current_tensor] = layer_index

            # lazy initialization of the cache
            just_allocated = False
            if self.__cache_tensors[self.__current_tensor] is None:
                cache_tensor = torch.zeros((self.__max_layer_bytes,), dtype=torch.int8, device=self.__device)
                if self.__is_pinned:
                    pin_tensor_(cache_tensor)
                self.__cache_tensors[self.__current_tensor] = cache_tensor
                just_allocated = True

            log(f"get_allocator {self.__current_tensor} for layer {layer_index}")

            return StaticTensorAllocator(self.__device, self.__cache_tensors[self.__current_tensor], just_allocated)
        else:
            return None


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


class LayerOffloadConductor:
    __module: nn.Module

    __layers: list[nn.Module]
    __layer_device_map: list[torch.device | None]
    __num_offloaded_layers: int
    __num_loaded_layers: int
    __offload_activations: bool
    __offload_layers: bool
    __layer_offload_fraction: float

    __layer_activations_included_offload_param_indices_map: list[list[int]]

    __train_device: torch.device
    __temp_device: torch.device

    __train_stream: torch.Stream | None
    __transfer_stream: torch.Stream | None

    __train_device_allocator: StaticLayerTensorAllocator
    __temp_device_allocator: StaticLayerTensorAllocator

    __layer_train_event_map: list[SyncEvent]
    __layer_transfer_event_map: list[SyncEvent]

    __activations_map: dict[int, Any]
    __activations_layer_index_map: dict[int, int]
    __activations_event_map: dict[int, SyncEvent]

    __is_forward_pass: bool
    __keep_graph: bool

    def __init__(
            self,
            module: nn.Module,
            config: TrainConfig,
    ):
        super(LayerOffloadConductor, self).__init__()

        self.__module = module

        self.__layers = []
        self.__layer_device_map = []
        self.__num_offloaded_layers = 0
        self.__num_loaded_layers = 0
        self.__offload_activations = config.gradient_checkpointing.offload()
        self.__offload_layers = config.gradient_checkpointing.offload() and config.layer_offload_fraction > 0
        self.__layer_offload_fraction = config.layer_offload_fraction

        self.__layer_activations_included_offload_param_indices_map = []

        self.__train_device = torch.device(config.train_device)
        self.__temp_device = torch.device(config.temp_device)

        self.__async_transfer = self.__train_device.type == "cuda"
        if self.__async_transfer:
            self.__train_stream = torch.cuda.default_stream(self.__train_device)
            self.__transfer_stream = torch.cuda.Stream(self.__train_device)
        else:
            self.__train_stream = None
            self.__transfer_stream = None

        self.__train_device_allocator = StaticLayerTensorAllocator(self.__train_device)
        self.__temp_device_allocator = StaticLayerTensorAllocator(self.__temp_device)

        self.__layer_train_event_map = []
        self.__layer_transfer_event_map = []

        self.__activations_map = {}
        self.__activations_layer_index_map = {}
        self.__activations_event_map = {}

        self.__is_forward_pass = False
        self.__keep_graph = False

    def offload_activated(self) -> bool:
        return self.__offload_activations or self.__offload_layers

    def layer_offload_activated(self) -> bool:
        return self.__offload_layers

    def to(self, device: torch.device):
        torch_gc()

        self.__init_layer_device_map()
        self.__wait_all_layer_transfers()

        if device_equals(device, self.__temp_device):
            log("to temp device")

            self.__module_to_device_except_layers(self.__temp_device)
            for layer_index, layer in enumerate(self.__layers):
                self.__layers[layer_index].to(self.__temp_device)
                self.__train_device_allocator.deallocate_cache(layer_index)
                self.__temp_device_allocator.deallocate_cache(layer_index)
                self.__layer_device_map[layer_index] = self.__temp_device

            self.__train_device_allocator.deallocate_cache()
            self.__temp_device_allocator.deallocate_cache()
        elif device_equals(device, self.__train_device):
            log("to train device")

            self.__train_device_allocator.allocate_cache(self.__layers, self.__num_loaded_layers + 1)
            self.__temp_device_allocator.allocate_cache(self.__layers, self.__num_offloaded_layers + 1)
            self.__module_to_device_except_layers(self.__train_device)

            # move all layers to the train device, then move offloadable tensors back to the temp device
            for layer_index, layer in enumerate(self.__layers):
                log(f"layer {layer_index} to train device")
                layer.to(self.__train_device)

                if layer_index < self.__num_loaded_layers:
                    allocator = self.__train_device_allocator.get_allocator(layer_index).allocate_like
                    for module in layer.modules():
                        offload_quantized(module, self.__train_device, allocator=allocator)
                    self.__layer_device_map[layer_index] = self.__train_device
                else:
                    allocator = self.__temp_device_allocator.get_allocator(layer_index).allocate_like
                    for module in layer.modules():
                        offload_quantized(module, self.__temp_device, allocator=allocator)
                    self.__layer_device_map[layer_index] = self.__temp_device

                event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
                self.__layer_train_event_map[layer_index] = event

        torch_gc()

    def add_layer(self, layer: nn.Module, included_offload_param_indices: list[int] = None):
        if included_offload_param_indices is None:
            included_offload_param_indices = []

        self.__layers.append(layer)
        self.__layer_device_map.append(None)
        self.__layer_train_event_map.append(SyncEvent())
        self.__layer_transfer_event_map.append(SyncEvent())
        self.__num_offloaded_layers = min(
            len(self.__layers) - 2,  # 2 layers need to be loaded for async offloading to work efficiently
            int(len(self.__layers) * self.__layer_offload_fraction)
        )
        self.__num_loaded_layers = len(self.__layers) - self.__num_offloaded_layers

        self.__layer_activations_included_offload_param_indices_map.append(included_offload_param_indices)

    def start_forward(self, keep_graph: bool):
        log()
        log()
        log()
        log("starting forward")
        self.__transfer_stream.wait_stream(self.__train_stream)
        self.__wait_all_layer_transfers()
        self.__init_layer_device_map()
        self.__clear_activations()

        self.__is_forward_pass = True
        self.__keep_graph = keep_graph

        if self.__offload_layers:
            for layer_index in range(len(self.__layers)):
                if layer_index < self.__num_loaded_layers:
                    self.__schedule_layer_to(layer_index, self.__train_device)
                else:
                    self.__schedule_layer_to(layer_index, self.__temp_device)

    def before_layer(self, layer_index: int, call_index: int, activations: Any) -> Any:
        log()
        log(f"before layer {layer_index}, {call_index}")

        if torch.is_grad_enabled() and self.__is_forward_pass:
            # Offloading can only be used with the use_reentrant=True checkpointing variant.
            # Gradients are only enabled during the back pass.
            log("starting backward")
            self.__is_forward_pass = False

        if call_index in self.__activations_map:
            replace_tensors_(
                activations, self.__activations_map.pop(call_index),
                self.__layer_activations_included_offload_param_indices_map[layer_index])

        self.__wait_layer_transfer(layer_index)

        if self.__offload_activations and not self.__is_forward_pass:
            # during the back pass, make sure activations are loaded before continuing
            self.__wait_all_activations_transfer()

            # if current activations are not on train_device, move them now
            if not tensors_match_device(
                    activations, self.__train_device,
                    self.__layer_activations_included_offload_param_indices_map[layer_index]):
                log(f"activations for layer {layer_index} not loaded to train device, transferring now")
                self.__schedule_activations_to_train_device(activations, self.__train_device, call_index)
                self.__wait_all_activations_transfer()

            # schedule previous activations to the train device
            if call_index - 1 in self.__activations_map:
                self.__schedule_activations_to_train_device(
                    self.__activations_map[call_index - 1], self.__train_device, call_index - 1)

        # schedule loading of the next layer and offloading of the previous layer
        if self.__offload_layers:  # TODO: implement a second offloading strategy: ring/toggle
            if self.__is_forward_pass and self.__keep_graph:
                # next pass will be a back pass.
                # do not offload the last layers, they will be needed immediately
                if 0 <= layer_index - 1 < self.__num_offloaded_layers:
                    self.__schedule_layer_to(layer_index - 1, self.__temp_device)
                if layer_index + self.__num_loaded_layers < len(self.__layers):
                    self.__schedule_layer_to(layer_index + self.__num_loaded_layers, self.__train_device)
            elif self.__is_forward_pass and not self.__keep_graph:
                # next pass will be another forward pass.
                # start loading the first layers when executing the last layers
                self.__schedule_layer_to(layer_index - 1, self.__temp_device)
                self.__schedule_layer_to(
                    (layer_index + self.__num_loaded_layers) % len(self.__layers), self.__train_device)
            elif not self.__is_forward_pass:
                # next pass will be a forward pass
                if self.__num_loaded_layers <= layer_index + 1 < len(self.__layers):
                    self.__schedule_layer_to(layer_index + 1, self.__temp_device)
                if layer_index - self.__num_loaded_layers >= 0:
                    self.__schedule_layer_to(layer_index - self.__num_loaded_layers, self.__train_device)

            return activations

    def after_layer(self, layer_index: int, call_index: int, activations: Any):
        log(f"after layer {layer_index}, {call_index}")

        # record stream
        if self.__async_transfer:
            for x in self.__get_all_tensors(self.__layers[layer_index]):
                x.record_stream(self.__train_stream)
        tensors_record_stream(self.__train_stream, activations)

        # save activations during the forward pass to make them accessible during the backward pass
        if self.__offload_activations and self.__keep_graph and self.__is_forward_pass:
            log(f"saving layer {call_index} activations for back pass")
            self.__activations_map[call_index] = activations
            self.__activations_layer_index_map[call_index] = layer_index
            self.__schedule_activations_to_temp_device(activations, self.__temp_device, layer_index, call_index)

        event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
        self.__layer_train_event_map[layer_index] = event

    def __module_to_device_except_layers(
            self,
            device: torch.device,
    ):
        sub_module_parameters = set(sum([list(x.parameters()) for x in self.__layers], []))

        def convert(t):
            if t in sub_module_parameters:
                return t

            return t.to(device=device)

        self.__module._apply(convert)

    @staticmethod
    def __get_all_tensors(layer: nn.Module):
        return sum([get_offload_tensors(x) for x in layer.modules()], [])

    def __init_layer_device_map(self):
        if self.__layer_device_map[0] is not None:
            return

        for layer_index, layer in enumerate(self.__layers):
            first_parameter_device = self.__get_all_tensors(layer)[0].device
            self.__layer_device_map[layer_index] = first_parameter_device

    def __clear_activations(self):
        self.__activations_map.clear()
        self.__activations_layer_index_map.clear()
        self.__activations_event_map.clear()

    def __wait_all_layer_train(self):
        for layer_index in range(len(self.__layers)):
            self.__wait_layer_train(layer_index)

    def __wait_all_layer_transfers(self):
        for layer_index in range(len(self.__layers)):
            self.__wait_layer_transfer(layer_index)

    def __wait_layer_train(self, layer_index: int):
        self.__layer_train_event_map[layer_index] \
            .wait(self.__transfer_stream, f"wait layer train {layer_index}")
        self.__layer_train_event_map[layer_index] = SyncEvent()

    def __wait_layer_transfer(self, layer_index: int):
        self.__layer_transfer_event_map[layer_index] \
            .wait(self.__train_stream, f"wait layer transfer {layer_index}")
        self.__layer_transfer_event_map[layer_index] = SyncEvent()

    def __wait_all_activations_transfer(self):
        for call_index, event in self.__activations_event_map.items():
            event.wait(self.__train_stream,
                       f"wait {call_index} activations layer {self.__activations_layer_index_map[call_index]}")
        self.__activations_event_map.clear()

    def __schedule_layer_to(
            self,
            layer_index: int,
            device: torch.device,
    ):
        current_device = self.__layer_device_map[layer_index]
        if device_equals(device, current_device):
            log(f"schedule layer {layer_index} to {str(device)}, skipping")
            return

        layer_allocator = self.__train_device_allocator \
            if device_equals(device, self.__train_device) \
            else self.__temp_device_allocator
        allocator = layer_allocator.get_allocator(layer_index)

        if allocator.just_allocated:
            event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
            self.__layer_train_event_map[layer_index] = event

        allocator_fn = allocator.allocate_like if allocator is not None else None

        with create_stream_context(self.__transfer_stream):
            self.__wait_layer_train(layer_index)
            layer = self.__layers[layer_index]
            if self.__async_transfer:
                parameters = self.__get_all_tensors(layer)
                parameter_pointers = [x.data_ptr() for x in parameters]
                log(f"layer {layer_index} pointers transfer: {parameter_pointers}")
                for module in layer.modules():
                    offload_quantized(module, device, non_blocking=True, allocator=allocator_fn)
                for x in parameters:
                    if x.device.type == "cuda":
                        x.record_stream(self.__transfer_stream)

                event = SyncEvent(self.__transfer_stream.record_event(), f"transfer to {device}")
                self.__layer_transfer_event_map[layer_index] = event
                log(f"schedule layer {layer_index} to {str(device)}, {event}")
            else:
                layer.to(device)
                log(f"schedule layer {layer_index} to {str(device)}, blocking")

            self.__layer_device_map[layer_index] = device

    def __schedule_activations_to_temp_device(
            self,
            activations: Any,
            device: torch.device,
            layer_index: int,
            call_index: int,
    ):
        log(f"schedule {call_index} activations to {str(device)}")

        with torch.cuda.stream(self.__transfer_stream):
            transferred = tensors_to_device_(
                activations, device,
                self.__layer_activations_included_offload_param_indices_map[layer_index], non_blocking=True)

            if transferred:
                tensors_record_stream(
                    self.__transfer_stream, activations,
                    self.__layer_activations_included_offload_param_indices_map[layer_index])
                self.__activations_event_map[call_index] = \
                    SyncEvent(self.__transfer_stream.record_event(), f"transfer to {device}")

    def __schedule_activations_to_train_device(
            self,
            activations: Any,
            device: torch.device,
            call_index: int,
    ):
        log(f"schedule {call_index} activations to {str(device)}")

        layer_index = self.__activations_layer_index_map[call_index]

        with torch.cuda.stream(self.__transfer_stream):
            transferred = tensors_to_device_(
                activations, device,
                self.__layer_activations_included_offload_param_indices_map[layer_index], non_blocking=True)

            if transferred:
                tensors_record_stream(
                    self.__transfer_stream, activations,
                    self.__layer_activations_included_offload_param_indices_map[layer_index])
                self.__activations_event_map[call_index] = \
                    SyncEvent(self.__transfer_stream.record_event(), f"transfer to {device}")
