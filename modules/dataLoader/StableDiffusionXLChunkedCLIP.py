from contextlib import nullcontext

from modules.model.util.clip_chunk import encode_clip_chunks, tokenize_clip_chunks

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class StableDiffusionXLChunkTokenize(PipelineModule, RandomAccessPipelineModule):
    def __init__(
            self,
            in_name: str,
            tokens_out_name: str,
            mask_out_name: str,
            tokenizer: CLIPTokenizer,
            token_limit: int,
    ):
        super().__init__()
        self.in_name = in_name
        self.tokens_out_name = tokens_out_name
        self.mask_out_name = mask_out_name
        self.tokenizer = tokenizer
        self.token_limit = token_limit

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.tokens_out_name, self.mask_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.in_name, index)
        tokens, mask = tokenize_clip_chunks(self.tokenizer, text, self.token_limit)
        return {
            self.tokens_out_name: tokens.to(self.pipeline.device),
            self.mask_out_name: mask.to(self.pipeline.device),
        }


class StableDiffusionXLChunkEncodeClipText(PipelineModule, RandomAccessPipelineModule):
    def __init__(
            self,
            in_name: str,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            hidden_state_output_index: int,
            autocast_contexts: list[torch.autocast | None] = None,
    ):
        super().__init__()
        self.in_name = in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.hidden_state_output_index = hidden_state_output_index
        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name] + ([self.pooled_out_name] if self.pooled_out_name else [])

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index).unsqueeze(0)
        with self._all_contexts(self.autocast_contexts):
            hidden_state, pooled_state = encode_clip_chunks(
                self.text_encoder,
                tokens,
                hidden_state_output_index=self.hidden_state_output_index,
                add_pooled_output=self.pooled_out_name is not None,
            )

        output = {self.hidden_state_out_name: hidden_state.squeeze(0)}
        if self.pooled_out_name:
            output[self.pooled_out_name] = pooled_state.squeeze(0)
        return output
