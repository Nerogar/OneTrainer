from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers.models.unets.unet_stable_cascade import SDCascadeAttnBlock, SDCascadeResBlock, SDCascadeTimestepBlock
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block


def create_checkpointed_forward(orig_module: nn.Module, device: torch.device) -> Callable:
    orig_forward = orig_module.forward

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
            dummy = torch.zeros((1,), device=device)
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


def enable_checkpointing_for_transformer_blocks(orig_module: nn.Module, device: torch.device):
    for _name, child_module in orig_module.named_modules():
        if isinstance(child_module, BasicTransformerBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)


def enable_checkpointing_for_clip_encoder_layers(orig_module: nn.Module, device: torch.device):
    for _name, child_module in orig_module.named_modules():
        if isinstance(child_module, CLIPEncoderLayer):
            child_module.forward = create_checkpointed_forward(child_module, device)


def enable_checkpointing_for_stable_cascade_blocks(orig_module: nn.Module, device: torch.device):
    for _name, child_module in orig_module.named_modules():
        if isinstance(child_module, SDCascadeResBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)
        if isinstance(child_module, SDCascadeAttnBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)
        if isinstance(child_module, SDCascadeTimestepBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)


def enable_checkpointing_for_t5_encoder_layers(orig_module: nn.Module, device: torch.device):
    for _name, child_module in orig_module.named_modules():
        if isinstance(child_module, T5Block):
            child_module.forward = create_checkpointed_forward(child_module, device)

def enable_checkpointing_for_stable_diffusion_3_transformer(orig_module: nn.Module, device: torch.device):
    for _name, child_module in orig_module.named_modules():
        if isinstance(child_module, JointTransformerBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)
