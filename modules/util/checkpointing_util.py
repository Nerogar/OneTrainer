import inspect
from collections.abc import Callable
from typing import Any

from modules.util.config.TrainConfig import TrainConfig
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformerBlock,
)
from diffusers.models.unets.unet_stable_cascade import SDCascadeAttnBlock, SDCascadeResBlock, SDCascadeTimestepBlock
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.t5.modeling_t5 import T5Block


def __kwargs_to_args(fun: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    signature = dict(inspect.signature(fun).parameters)
    parameters = []

    for i, (key, value) in enumerate(signature.items()):
        if i < len(args):
            parameters.append(args[i])
        elif key in kwargs:
            parameters.append(kwargs[key])
        elif value.default is not value.empty:
            parameters.append(value.default)
        else:
            raise Exception("could not convert from kwargs to args")

    return tuple(parameters)


def __get_args_indices(fun: Callable, arg_names: list[str]) -> list[int]:
    signature = dict(inspect.signature(fun).parameters)
    indices = []

    for i, key in enumerate(signature.keys()):
        if key in arg_names:
            indices.append(i)

    return indices


__current_call_index = 0


def __generate_call_index() -> int:
    global __current_call_index
    __current_call_index += 1
    return __current_call_index


def create_checkpointed_forward(
        orig_module: nn.Module,
        train_device: torch.device,
        include_from_offload_param_names: list[str] = None,
        conductor: LayerOffloadConductor | None = None,
        layer_index: int = 0,
) -> Callable:
    orig_forward = orig_module.forward
    if include_from_offload_param_names is None:
        include_from_offload_param_names = []
    included_offload_param_indices = __get_args_indices(orig_forward, include_from_offload_param_names)

    bound_conductor = conductor
    bound_layer_index = layer_index
    if conductor is not None:
        conductor.add_layer(orig_module, included_offload_param_indices)

    if conductor is not None and conductor.offload_activated():
        def offloaded_custom_forward(
                # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
                dummy: torch.Tensor,
                call_id: int,
                *args,
        ):
            if bound_layer_index == 0 and not torch.is_grad_enabled():
                bound_conductor.start_forward(True)

            args = bound_conductor.before_layer(bound_layer_index, call_id, args)
            output = orig_forward(*args)
            bound_conductor.after_layer(bound_layer_index, call_id, args)
            return output

        def custom_forward(
                # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
                dummy: torch.Tensor,
                call_index: int,
                *args,
        ):
            if bound_layer_index == 0:
                bound_conductor.start_forward(False)

            args = bound_conductor.before_layer(bound_layer_index, call_index, args)
            output = orig_forward(*args)
            bound_conductor.after_layer(bound_layer_index, call_index, args)
            return output

        def forward(
                *args,
                **kwargs
        ):
            call_id = __generate_call_index()

            if torch.is_grad_enabled():
                dummy = torch.zeros((1,), device=train_device)
                dummy.requires_grad_(True)

                args = __kwargs_to_args(orig_forward, args, kwargs)

                return checkpoint(
                    offloaded_custom_forward,
                    dummy,
                    call_id,
                    *args,
                    use_reentrant=True
                )
            else:
                args = __kwargs_to_args(orig_forward, args, kwargs)
                return custom_forward(None, call_id, *args)
    else:
        def custom_forward(
                # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
                dummy: torch.Tensor = None,
                *args,
                **kwargs,
        ):
            return orig_forward(
                *args,
                **kwargs,
            )

        def forward(
                *args,
                **kwargs
        ):
            if torch.is_grad_enabled():
                dummy = torch.zeros((1,), device=train_device)
                dummy.requires_grad_(True)

                return checkpoint(
                    custom_forward,
                    dummy,
                    *args,
                    **kwargs,
                    use_reentrant=False
                )
            else:
                return custom_forward(None, *args, **kwargs)

    return forward


def enable_checkpointing_for_basic_transformer_blocks(
        orig_module: nn.Module,
        config: TrainConfig,
        offload_enabled: bool,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, BasicTransformerBlock):
            if offload_enabled:
                child_module.forward = create_checkpointed_forward(
                    child_module, torch.device(config.train_device),
                    [],
                    conductor, layer_index,
                )
            else:
                child_module.forward = create_checkpointed_forward(
                    child_module, torch.device(config.train_device),
                    [],
                )
            layer_index += 1

    return conductor


def enable_checkpointing_for_clip_encoder_layers(
        orig_module: nn.Module,
        config: TrainConfig,
):
    for child_module in orig_module.modules():
        if isinstance(child_module, CLIPEncoderLayer):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                [],
            )


def enable_checkpointing_for_stable_cascade_blocks(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, SDCascadeResBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                [],
                conductor, layer_index,
            )
            layer_index += 1
        if isinstance(child_module, SDCascadeAttnBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                [],
                conductor, layer_index,
            )
            layer_index += 1
        if isinstance(child_module, SDCascadeTimestepBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_t5_encoder_layers(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, T5Block):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_gemma_layers(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, Gemma2DecoderLayer):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_llama_encoder_layers(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, LlamaDecoderLayer):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_stable_diffusion_3_transformer(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, JointTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_flux_transformer(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, FluxTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states", "encoder_hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    for child_module in orig_module.modules():
        if isinstance(child_module, FluxSingleTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor

def enable_checkpointing_for_sana_transformer(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, SanaTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor

def enable_checkpointing_for_hunyuan_video_transformer(
        orig_module: nn.Module,
        config: TrainConfig,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(orig_module, config)

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, HunyuanVideoTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states", "encoder_hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    for child_module in orig_module.modules():
        if isinstance(child_module, HunyuanVideoSingleTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, torch.device(config.train_device),
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor
