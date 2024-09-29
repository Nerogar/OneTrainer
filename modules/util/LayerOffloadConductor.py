import random
from typing import Any

import torch
from torch import nn

from modules.util.config.TrainConfig import TrainConfig
from modules.util.quantization_util import offload_quantized, get_offload_tensors
from modules.util.torch_util import (
    create_stream_context,
    device_equals,
    tensor_to_device_, module_to_device_except_sub_module,
)

MESSAGES = []


def log(msg: str = ''):
    pass
    print(msg)
    # MESSAGES.append(msg)


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

    __layer_train_event_map: list[SyncEvent]
    __layer_transfer_event_map: list[SyncEvent]

    __activations_list: list[Any]
    __activations_event_list: list[SyncEvent]

    __train_stream: torch.Stream | None
    __transfer_stream: torch.Stream | None

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

        self.__layer_train_event_map = []
        self.__layer_transfer_event_map = []

        self.__activations_list = []
        self.__activations_event_list = []

        self.__async_transfer = self.__train_device.type == "cuda"
        if self.__async_transfer:
            self.__train_stream = torch.cuda.default_stream(self.__train_device)
            self.__transfer_stream = torch.cuda.Stream(self.__train_device)
        else:
            self.__train_stream = None
            self.__transfer_stream = None

        self.__is_forward_pass = False
        self.__keep_graph = False

    def offload_activated(self) -> bool:
        return self.__offload_activations or self.__offload_layers

    def layer_offload_activated(self) -> bool:
        return self.__offload_layers

    def to(self, device: torch.device):
        self.__wait_all_layer_transfers()

        if device_equals(device, self.__temp_device):
            log("to temp device")
            self.__module.to(self.__temp_device)
        elif device_equals(device, self.__train_device):
            log("to train device")
            module_to_device_except_sub_module(self.__module, device, self.__layers)

            # move all layers to the train device, then move offloadable tensors back to the temp device
            for layer_index, layer in enumerate(self.__layers):
                log(f"layer {layer_index} to train device")
                layer.to(self.__train_device)
                for module in layer.modules():
                    offload_quantized(module, self.__temp_device)

    def add_layer(self, layer: nn.Module, included_offload_param_indices: list[int] = None):
        if included_offload_param_indices is None:
            included_offload_param_indices = []

        self.__layers.append(layer)
        self.__layer_device_map.append(None)
        self.__layer_train_event_map.append(SyncEvent())
        self.__layer_transfer_event_map.append(SyncEvent())
        self.__num_offloaded_layers = int(len(self.__layers) * self.__layer_offload_fraction)
        self.__num_loaded_layers = len(self.__layers) - self.__num_offloaded_layers

        self.__layer_activations_included_offload_param_indices_map.append(included_offload_param_indices)

    def start_forward(self, keep_graph: bool):
        log()
        log()
        log()
        log("starting forward")
        # torch.cuda.synchronize(self.__train_device)
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

    def before_layer(self, layer_index: int):
        log()
        log(f"before layer {layer_index}")
        if torch.is_grad_enabled() and self.__is_forward_pass:
            # Offloading can only be used with the use_reentrant=True checkpointing variant.
            # Gradients are only enabled during the back pass.
            log("starting backward")
            self.__is_forward_pass = False

        self.__wait_layer_transfer(layer_index)

        if self.__offload_activations:
            # during the back pass, make sure activations are loaded before continuing
            if not self.__is_forward_pass:
                # schedule load of the previous activations
                previous_activations_index = len(self.__activations_list) - 2
                if previous_activations_index >= 0:
                    previous_activations_event = self.__schedule_activations_to(
                        self.__activations_list[previous_activations_index], self.__train_device)
                    self.__activations_event_list[previous_activations_index] = previous_activations_event

                # wait for activations to be loaded
                activations = self.__activations_list.pop()
                activations_event = self.__activations_event_list.pop()
                if activations_event is None:  # if move was never scheduled, schedule it now
                    log(f"activations not scheduled, scheduling now to {str(self.__train_device)}")
                    activations_event = self.__schedule_activations_to(activations, self.__train_device)
                activations_event.synchronize()

        # schedule loading of the next layer and offloading of the previous layer
        if self.__offload_layers:
            if self.__is_forward_pass and self.__keep_graph:
                # next pass will be a back pass.
                # do not offload the last layers, they will be needed immediately
                if layer_index + self.__num_loaded_layers < len(self.__layers):
                    self.__schedule_layer_to(layer_index + self.__num_loaded_layers, self.__train_device)
                if 0 <= layer_index - 1 < self.__num_offloaded_layers:
                    self.__schedule_layer_to(layer_index - 1, self.__temp_device)
            elif self.__is_forward_pass and not self.__keep_graph:
                # next pass will likely be another forward pass.
                # start loading the first layers when executing the last layers
                self.__schedule_layer_to(
                    (layer_index + self.__num_loaded_layers) % len(self.__layers), self.__train_device)
                self.__schedule_layer_to(layer_index - 1, self.__temp_device)
            elif not self.__is_forward_pass:
                # next pass will be a forward pass
                if layer_index - self.__num_loaded_layers >= 0:
                    self.__schedule_layer_to(layer_index - self.__num_loaded_layers, self.__train_device)
                if self.__num_loaded_layers <= layer_index + 1 < len(self.__layers):
                    self.__schedule_layer_to(layer_index + 1, self.__temp_device)

    def after_layer(self, layer_index: int, activations: Any):
        log(f"after layer {layer_index}")

        if self.__async_transfer:
            for x in self.__get_all_tensors(self.__layers[layer_index]):
                x.record_stream(self.__train_stream)

        if self.__offload_activations and self.__keep_graph and self.__is_forward_pass:
            # save activations during the forward pass to make them accessible during the backward pass
            self.__activations_list.append(activations)
            self.__activations_event_list.append(self.__schedule_activations_to(activations, self.__temp_device))

        event = SyncEvent(self.__train_stream.record_event(), f"train on {self.__train_device}")
        self.__layer_train_event_map[layer_index] = event

    def __get_all_tensors(self, layer: nn.Module):
        return sum([get_offload_tensors(x) for x in layer.modules()], [])

    def __init_layer_device_map(self):
        for layer_index, layer in enumerate(self.__layers):
            first_parameter_device = self.__get_all_tensors(layer)[0].device
            self.__layer_device_map[layer_index] = first_parameter_device

    def __clear_activations(self):
        self.__activations_list = []
        self.__activations_event_list = []

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

    def __schedule_layer_to(
            self,
            layer_index: int,
            device: torch.device,
    ):
        current_device = self.__layer_device_map[layer_index]
        if device_equals(device, current_device):
            log(f"schedule layer {layer_index} to {str(device)}, skipping")
            return

        with create_stream_context(self.__transfer_stream):
            self.__wait_layer_train(layer_index)
            layer = self.__layers[layer_index]
            if self.__async_transfer:
                parameters = self.__get_all_tensors(layer)
                parameter_pointers = [x.data_ptr() for x in parameters]
                log(f"layer {layer_index} pointers transfer: {parameter_pointers}")
                for module in layer.modules():
                    offload_quantized(module, device, non_blocking=True)
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

    def __schedule_activations_to(
            self,
            activations: Any,
            device: torch.device,
    ) -> SyncEvent:
        log(f"schedule activations to {str(device)}")

        with torch.cuda.stream(self.__transfer_stream):
            tensor_to_device_(activations, device, non_blocking=True)
            return SyncEvent(self.__transfer_stream.record_event(), f"transfer to {device}")
