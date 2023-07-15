from typing import Callable, Optional, Dict, Any

import torch
from diffusers.models.attention import BasicTransformerBlock
from torch import nn


def __create_basic_transformer_block_forward(orig_module) -> Callable:
    orig_forward = orig_module.forward

    def forward(
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        return torch.utils.checkpoint.checkpoint(
            orig_forward,
            hidden_states,  # hidden_states
            attention_mask,  # attention_mask
            encoder_hidden_states,  # encoder_hidden_states
            encoder_attention_mask,  # encoder_attention_mask
            timestep,  # timestep
            cross_attention_kwargs,  # cross_attention_kwargs
            class_labels,  # class_labels
            use_reentrant=False
        )

    return forward


def enable_checkpointing_for_transformer_blocks(orig_module: nn.Module):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, BasicTransformerBlock):
            child_module.forward = __create_basic_transformer_block_forward(child_module)
