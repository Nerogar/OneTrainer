import inspect
from typing import Callable, Any

import torch
from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.models.unets.unet_stable_cascade import SDCascadeAttnBlock, SDCascadeResBlock, SDCascadeTimestepBlock
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
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

    for i, (key, value) in enumerate(signature.items()):
        if key in arg_names:
            indices.append(i)

    return indices


def to_(
        data: torch.Tensor | list | tuple | dict,
        device: torch.device,
        include_parameter_indices: list[int] = None,
):
    if include_parameter_indices is not None and len(include_parameter_indices) == 0:
        include_parameter_indices = None

    if isinstance(data, torch.Tensor):
        data.data = data.data.to(device=device)
    elif isinstance(data, (list, tuple)):
        for i, elem in enumerate(data):
            if include_parameter_indices is None or i in include_parameter_indices:
                to_(elem, device)
    elif isinstance(data, dict):
        for elem in data.values():
            to_(elem, device)


def to(data: torch.Tensor | list | tuple | dict, device: torch.device) -> torch.Tensor | list | tuple | dict:
    if isinstance(data, torch.Tensor):
        return data.to(device=device)
    elif isinstance(data, (list, tuple)):
        for i in range(len(data)):
            data[i] = to(data[i], device)
    elif isinstance(data, dict):
        for key, elem in data.items():
            data[key] = to(elem, device)


def create_checkpointed_forward(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload_activations: bool = False,
        include_from_offload_param_names: list[str] = None
) -> Callable:
    orig_forward = orig_module.forward
    if include_from_offload_param_names is None:
        include_from_offload_param_names = []
    include_from_offload_param_indices = __get_args_indices(orig_forward, include_from_offload_param_names)

    if offload_activations:
        def custom_forward(
                # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
                dummy: torch.Tensor = None,
                *args,
        ):
            to_(args, train_device, include_from_offload_param_indices)

            output = orig_forward(*args)

            if not torch.is_grad_enabled():
                to_(args, temp_device, include_from_offload_param_indices)

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
                    custom_forward,
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
        offload_activations: bool = True,
):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, BasicTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )


def enable_checkpointing_for_clip_encoder_layers(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload_activations: bool = True,
):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, CLIPEncoderLayer):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )


def enable_checkpointing_for_stable_cascade_blocks(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload_activations: bool = True,
):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, SDCascadeResBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )
        if isinstance(child_module, SDCascadeAttnBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )
        if isinstance(child_module, SDCascadeTimestepBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )


def enable_checkpointing_for_t5_encoder_layers(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload_activations: bool = True,
):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, T5Block):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )


def enable_checkpointing_for_stable_diffusion_3_transformer(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload_activations: bool = True,
):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, JointTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, [],
            )


def enable_checkpointing_for_flux_transformer(
        orig_module: nn.Module,
        train_device: torch.device,
        temp_device: torch.device,
        offload_activations: bool = True,
):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, FluxTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, ["hidden_states", "encoder_hidden_states"],
            )
        if isinstance(child_module, FluxSingleTransformerBlock):
            child_module.forward = create_checkpointed_forward(
                child_module, train_device, temp_device,
                offload_activations, ["hidden_states"],
            )
