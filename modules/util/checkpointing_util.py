import inspect
from collections.abc import Callable
from typing import Any

from modules.util.config.TrainConfig import TrainConfig
from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.torch_util import add_dummy_grad_fn_, has_grad_fn

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

torch._dynamo.config.cache_size_limit = 8192

def _kwargs_to_args(fun: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    signature = dict(inspect.signature(fun).parameters)
    parameters = []

    for i, (key, value) in enumerate(signature.items()):
        if i < len(args):
            parameters.append(args[i])
        elif key in kwargs:
            parameters.append(kwargs[key])
        elif value.default is not value.empty:
            parameters.append(value.default)

    return tuple(parameters)


def __get_args_indices(fun: Callable, arg_names: list[str]) -> list[int]:
    signature = dict(inspect.signature(fun).parameters)
    indices = []

    for i, key in enumerate(signature.keys()):
        if key in arg_names:
            indices.append(i)

    return indices


__current_call_index = 0


def _generate_call_index() -> int:
    global __current_call_index
    __current_call_index += 1
    return __current_call_index


class CheckpointLayer(torch.nn.Module):
    def __init__(self, orig: nn.Module, train_device: torch.device):
        super().__init__()
        self.checkpoint = orig
        # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
        self.dummy = torch.zeros((1,), device=train_device, requires_grad=True)

    def __checkpointing_forward(self, dummy: torch.Tensor, *args, **kwargs):
        return self.checkpoint(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(
                self.__checkpointing_forward,
                self.dummy,
                *args,
                **kwargs,
                use_reentrant=False
            )
        else:
            return self.checkpoint(*args, **kwargs)

class OffloadCheckpointLayer(torch.nn.Module):
    def __init__(self, orig: nn.Module, train_device: torch.device, conductor: LayerOffloadConductor, layer_index: int):
        super().__init__()
        self.checkpoint = orig
        self.dummy = torch.zeros((1,), device=train_device, requires_grad=True)
        self.conductor = conductor
        self.layer_index = layer_index

    def __checkpointing_forward(self, dummy: torch.Tensor, call_id: int, *args):

        if self.layer_index == 0 and not torch.is_grad_enabled():
            self.conductor.start_forward(True)

        args = self.conductor.before_layer(self.layer_index, call_id, args)
        output = self.checkpoint(*args)
        self.conductor.after_layer(self.layer_index, call_id, args)

        # make sure at least one of the output tensors has a grad_fn so the output of the checkpoint has a grad_fn
        assert not (torch.is_grad_enabled() and not has_grad_fn(output))
        #TODO how can this be the case? Is there a backward that does not produce gradients wrt to any of its inputs?
        #if it be the case, TODO check that add_dummy_grad_fn_ still works with torch.compile
        if torch.is_grad_enabled() and not has_grad_fn(output):
            output = add_dummy_grad_fn_(output)

        return output

    def forward(self, *args, **kwargs):
        call_id = _generate_call_index()
        args = _kwargs_to_args(self.checkpoint.forward, args, kwargs)

        if torch.is_grad_enabled():
            return checkpoint(
                self.__checkpointing_forward,
                self.dummy,
                call_id,
                *args,
                use_reentrant=True
            )
        else:
            if self.layer_index == 0:
                self.conductor.start_forward(False)

            args = self.conductor.before_layer(self.layer_index, call_id, args)
            output = self.checkpoint(*args)
            self.conductor.after_layer(self.layer_index, call_id, args)
            return output

def create_checkpoint(
        orig_module: nn.Module,
        train_device: torch.device,
        include_from_offload_param_names: list[str] = None,
        conductor: LayerOffloadConductor | None = None,
        layer_index: int = 0,
        compile: bool = False,
        enabled: bool = True,
) -> Callable:
    if include_from_offload_param_names is None:
        include_from_offload_param_names = []
    included_offload_param_indices = __get_args_indices(orig_module.forward, include_from_offload_param_names)

    if conductor is not None:
        conductor.add_layer(orig_module, included_offload_param_indices)

    if conductor is not None and conductor.offload_activated():
        layer = OffloadCheckpointLayer(orig_module, train_device, conductor, layer_index)
        if compile:
            #don't compile the checkpointing layer - offloading cannot be compiled:
            orig_module.compile(fullgraph=True)
    else:
        layer = CheckpointLayer(orig_module, train_device) if enabled else orig_module
        if compile:
            #do compile the checkpointing layer - slightly faster
            layer.compile(fullgraph=True)
    return layer

def _create_checkpoints_for_module_list(
        module_list: nn.ModuleList,
        include_from_offload_param_names: list[str],
        conductor: LayerOffloadConductor,
        train_device: torch.device,
        layer_index: int,
        compile: bool,
) -> int:

    for i, layer in enumerate(module_list):
        module_list[i] = create_checkpoint(
                layer, train_device,
                include_from_offload_param_names,
                conductor, layer_index, compile=compile,
            )
        layer_index += 1
    return layer_index


def enable_checkpointing(
        model: nn.Module,
        config: TrainConfig,
        compile: bool,
        lists,
        offload_enabled: bool = True,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(model, config)

    layer_index = 0
    for module_list, param_names in lists:
        layer_index = _create_checkpoints_for_module_list(
            module_list,
            param_names,
            conductor if offload_enabled else None,
            torch.device(config.train_device),
            layer_index,
            compile = compile,
        )

    return conductor


#TODO test all models
def enable_checkpointing_for_basic_transformer_blocks(
        model: nn.Module,
        config: TrainConfig,
        offload_enabled: bool,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
            (model.transformer_blocks,        []),
        ],
        offload_enabled = offload_enabled,
    )

def enable_checkpointing_for_clip_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
):
    return enable_checkpointing(model, config, False, [
        (model.text_model.encoder.layers, []), # No activation offloading for text encoders, because the output might be taken from the middle of the network
    ])

def enable_checkpointing_for_stable_cascade_blocks(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.down_blocks, []),
        (model.up_blocks, []),
    ])

def enable_checkpointing_for_t5_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, False, [
        (model.encoder.block, []),
    ])


def enable_checkpointing_for_gemma_layers(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, False, [
        (model.layers, []),
    ])


def enable_checkpointing_for_llama_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, False, [
        (model.model.layers, []),
    ])

def enable_checkpointing_for_qwen_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, False, [
        (model.model.language_model.layers, []),  # TODO No activation offloading for other encoders, see above. But clip skip is not implemented for QwenVL. Then do activation offloading?
    ])

def enable_checkpointing_for_stable_diffusion_3_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"]),
    ])

def enable_checkpointing_for_flux_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"                         ]),
    ])


def enable_checkpointing_for_chroma_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"                         ]),
    ])


def enable_checkpointing_for_qwen_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.transformer_blocks, ["hidden_states", "encoder_hidden_states"]),
    ])


def enable_checkpointing_for_sana_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.transformer_blocks, ["hidden_states"]),
    ])

def enable_checkpointing_for_hunyuan_video_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.context_embedder.token_refiner.refiner_blocks, ["hidden_states"                         ]),
        (model.transformer_blocks,                            ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks,                     ["hidden_states"                         ]),
    ])

def enable_checkpointing_for_hi_dream_transformer(
        model: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    return enable_checkpointing(model, config, config.compile, [
        (model.double_stream_blocks, ["hidden_states", "encoder_hidden_states"]),
        (model.single_stream_blocks, ["hidden_states"                         ]),
    ])
