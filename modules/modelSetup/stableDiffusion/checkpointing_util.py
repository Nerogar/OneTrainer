from typing import Callable

import torch
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unets.unet_stable_cascade import SDCascadeTimestepBlock, SDCascadeAttnBlock, SDCascadeResBlock
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


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
        dummy = torch.zeros((1,), device=device)
        dummy.requires_grad_(True)

        return checkpoint(
            custom_forward,
            dummy,
            *args,
            **kwargs,
            use_reentrant=False
        )

    return forward


def enable_checkpointing_for_transformer_blocks(orig_module: nn.Module, device: torch.device):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, BasicTransformerBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)


def enable_checkpointing_for_clip_encoder_layers(orig_module: nn.Module, device: torch.device):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, CLIPEncoderLayer):
            child_module.forward = create_checkpointed_forward(child_module, device)


def enable_checkpointing_for_stable_cascade_blocks(orig_module: nn.Module, device: torch.device):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, SDCascadeResBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)
        if isinstance(child_module, SDCascadeAttnBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)
        if isinstance(child_module, SDCascadeTimestepBlock):
            child_module.forward = create_checkpointed_forward(child_module, device)
