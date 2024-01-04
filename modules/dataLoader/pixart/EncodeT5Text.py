from contextlib import nullcontext

import torch
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from transformers import T5EncoderModel


class EncodeT5Text(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            tokens_in_name: str,
            tokens_attention_mask_in_name: str,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: T5EncoderModel,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_context: torch.autocast | None = None,
    ):
        super(EncodeT5Text, self).__init__()
        self.tokens_in_name = tokens_in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = hidden_state_output_index

        self.autocast_context = nullcontext() if autocast_context is None else autocast_context
        self.autocast_enabled = isinstance(self.autocast_context, torch.autocast)

    def length(self) -> int:
        return self._get_previous_length(self.tokens_in_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_in_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        if self.pooled_out_name:
            return [self.hidden_state_out_name, self.pooled_out_name]
        else:
            return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_in_name, index)
        tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)

        tokens = tokens.unsqueeze(0)
        tokens_attention_mask = tokens_attention_mask.unsqueeze(0)

        with self.autocast_context:
            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = text_encoder_output.hidden_states[:-1]
        if self.pooled_out_name:
            pooled_state = text_encoder_output.text_embeds
        else:
            pooled_state = None

        hidden_states = [hidden_state.squeeze() for hidden_state in hidden_states]
        pooled_state = None if pooled_state is None else pooled_state.squeeze()

        hidden_state = hidden_states[self.hidden_state_output_index]

        if self.add_layer_norm:
            final_layer_norm = self.text_encoder.encoder.final_layer_norm
            hidden_state = final_layer_norm(
                hidden_state
            )

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }
