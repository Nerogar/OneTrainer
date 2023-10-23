from typing import Callable, Optional, Dict, Any, Union, Tuple

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from torch import nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


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
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            cross_attention_kwargs,
            class_labels,
            use_reentrant=False
        )

    return forward


def enable_checkpointing_for_transformer_blocks(orig_module: nn.Module):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, BasicTransformerBlock):
            child_module.forward = __create_basic_transformer_block_forward(child_module)


def __create_clip_encoder_layer_forward(orig_module) -> Callable:
    orig_forward = orig_module.forward

    def custom_forward(
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
            # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
            dummy: torch.Tensor = None,
    ):
        return orig_forward(
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions,
        )

    def forward(
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ):
        dummy = torch.zeros((1,), device=hidden_states.device)
        dummy.requires_grad_(True)

        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            hidden_states,
            attention_mask,
            causal_attention_mask,
            output_attentions,
            dummy,
            use_reentrant=False
        )

    return forward


def enable_checkpointing_for_clip_encoder_layers(orig_module: nn.Module):
    for name, child_module in orig_module.named_modules():
        if isinstance(child_module, CLIPEncoderLayer):
            child_module.forward = __create_clip_encoder_layer_forward(child_module)

def create_checkpointed_unet_forward(orig_module) -> Callable:
    orig_forward = orig_module.forward

    def custom_forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
            dummy: torch.Tensor = None,
    ):
        return orig_forward(
            sample,
            timestep,
            encoder_hidden_states,
            class_labels,
            timestep_cond,
            attention_mask,
            cross_attention_kwargs,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
            encoder_attention_mask,
            return_dict,
        )

    def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        dummy = torch.zeros((1,), device=sample.device)
        dummy.requires_grad_(True)

        return torch.utils.checkpoint.checkpoint(
            custom_forward,
            sample,
            timestep,
            encoder_hidden_states,
            class_labels,
            timestep_cond,
            attention_mask,
            cross_attention_kwargs,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
            encoder_attention_mask,
            return_dict,
            dummy,
            use_reentrant=False
        )

    return forward