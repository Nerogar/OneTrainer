import inspect
from typing import Any, Callable

import torch
from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.models.unets.unet_stable_cascade import SDCascadeAttnBlock, SDCascadeResBlock, SDCascadeTimestepBlock
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block

from modules.util.LayerOffloadConductor import LayerOffloadConductor


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
    conductor.add_layer(orig_module, included_offload_param_indices)

    if conductor is not None and conductor.offload_activated():
        def offloaded_custom_forward(
                # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
                dummy: torch.Tensor = None,
                *args,
        ):
            if bound_layer_index == 0 and not torch.is_grad_enabled():
                bound_conductor.start_forward(True)

            bound_conductor.before_layer(bound_layer_index)
            output = orig_forward(*args)
            bound_conductor.after_layer(bound_layer_index, args)
            return output

        def custom_forward(
                # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
                dummy: torch.Tensor = None,
                *args,
        ):
            if bound_layer_index == 0:
                bound_conductor.start_forward(False)

            bound_conductor.before_layer(bound_layer_index)
            output = orig_forward(*args)
            bound_conductor.after_layer(bound_layer_index, args)
            return output

        def forward(
                *args,
                **kwargs
        ):
            if torch.is_grad_enabled():
                dummy = torch.zeros((1,), device=train_device)
                dummy.requires_grad_(True)

                args = __kwargs_to_args(orig_forward, args, kwargs)

                return checkpoint(
                    offloaded_custom_forward,
                    dummy,
                    *args,
                    use_reentrant=True
                )
            else:
                args = __kwargs_to_args(orig_forward, args, kwargs)
                return custom_forward(None, *args)
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


def enable_checkpointing_for_sdxl_transformer_blocks(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload: bool = False,
        layer_offload_fraction: float = 0.0,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(
        orig_module,
        train_device,
        temp_device,
        offload_activations=offload,
        offload_layers=offload and layer_offload_fraction > 0,
        layer_offload_fraction=layer_offload_fraction,
    )

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, BasicTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_clip_encoder_layers(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload: bool = False,
        layer_offload_fraction: float = 0.0,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(
        orig_module,
        train_device,
        temp_device,
        offload_activations=offload,
        offload_layers=offload and layer_offload_fraction > 0,
        layer_offload_fraction=layer_offload_fraction,
    )

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, CLIPEncoderLayer):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_stable_cascade_blocks(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload: bool = False,
        layer_offload_fraction: float = 0.0,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(
        orig_module,
        train_device,
        temp_device,
        offload_activations=offload,
        offload_layers=offload and layer_offload_fraction > 0,
        layer_offload_fraction=layer_offload_fraction,
    )

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, SDCascadeResBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1
        if isinstance(child_module, SDCascadeAttnBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1
        if isinstance(child_module, SDCascadeTimestepBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_t5_encoder_layers(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload: bool = False,
        layer_offload_fraction: float = 0.0,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(
        orig_module,
        train_device,
        temp_device,
        offload_activations=offload,
        offload_layers=offload and layer_offload_fraction > 0,
        layer_offload_fraction=layer_offload_fraction,
    )

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, T5Block):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_stable_diffusion_3_transformer(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload: bool = False,
        layer_offload_fraction: float = 0.0,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(
        orig_module,
        train_device,
        temp_device,
        offload_activations=offload,
        offload_layers=offload and layer_offload_fraction > 0,
        layer_offload_fraction=layer_offload_fraction,
    )

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, JointTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                [],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor


def enable_checkpointing_for_flux_transformer(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload: bool = False,
        layer_offload_fraction: float = 0.0,
) -> LayerOffloadConductor:
    conductor = LayerOffloadConductor(
        orig_module,
        train_device,
        temp_device,
        offload_activations=offload,
        offload_layers=offload and layer_offload_fraction > 0,
        layer_offload_fraction=layer_offload_fraction,
    )

    layer_index = 0
    for child_module in orig_module.modules():
        if isinstance(child_module, FluxTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                ["hidden_states", "encoder_hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    for child_module in orig_module.modules():
        if isinstance(child_module, FluxSingleTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device,
                ["hidden_states"],
                conductor, layer_index,
            )
            layer_index += 1

    return conductor
