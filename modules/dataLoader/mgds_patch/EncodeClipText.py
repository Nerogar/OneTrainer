from contextlib import nullcontext
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

from transformers import CLIPTextModel, CLIPTextModelWithProjection


class EncodeClipText(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            pooled_out_name: str | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            add_layer_norm: bool,
            hidden_state_output_index: int | None = None,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
            chunk_if_needed: bool = False,
            pooled_output_handling: str = 'FIRST',
            tokenizer: Any | None = None,
            split_on_comma: bool = False,
            max_chunks: int | None = None,
    ):
        super().__init__()
        self.in_name = in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.pooled_out_name = pooled_out_name
        self.text_encoder = text_encoder
        self.add_layer_norm = add_layer_norm
        self.hidden_state_output_index = -1 if hidden_state_output_index is None else hidden_state_output_index

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

        self.chunk_if_needed = chunk_if_needed
        self.chunk_size = text_encoder.config.max_position_embeddings - 2
        self.pooled_output_handling = pooled_output_handling
        self.tokenizer = tokenizer
        self.split_on_comma = split_on_comma
        self.max_chunks = max_chunks

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        if self.pooled_out_name:
            return [self.hidden_state_out_name, self.pooled_out_name]
        else:
            return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.in_name, index)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(variation, self.tokens_attention_mask_in_name, index)
        else:
            tokens_attention_mask = None

        if self.chunk_if_needed and tokens.shape[0] > self.chunk_size + 2:
            return self._get_item_chunked(tokens, tokens_attention_mask, self.chunk_size)
        else:
            return self._get_item_single(tokens, tokens_attention_mask)

    def _get_item_single(self, tokens: torch.Tensor, tokens_attention_mask: torch.Tensor | None) -> dict:
        tokens = tokens.unsqueeze(0)
        if tokens_attention_mask is not None:
            tokens_attention_mask = tokens_attention_mask.unsqueeze(0)

        with self._all_contexts(self.autocast_contexts):
            if tokens_attention_mask is not None and self.dtype:
                tokens_attention_mask = tokens_attention_mask.to(dtype=self.dtype)

            text_encoder_output = self.text_encoder(
                tokens,
                attention_mask=tokens_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = text_encoder_output.hidden_states
        if self.pooled_out_name:
            pooled_state = None
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_state = text_encoder_output.text_embeds
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_state = text_encoder_output.pooler_output
        else:
            pooled_state = None

        hidden_states = [hidden_state.squeeze(dim=0) for hidden_state in hidden_states]
        pooled_state = None if pooled_state is None else pooled_state.squeeze(dim=0)

        hidden_state = hidden_states[self.hidden_state_output_index]

        if self.add_layer_norm:
            with self._all_contexts(self.autocast_contexts):
                final_layer_norm = self.text_encoder.text_model.final_layer_norm
                hidden_state = final_layer_norm(
                    hidden_state
                )

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }

    def _get_item_chunked(self, tokens: torch.Tensor, tokens_attention_mask: torch.Tensor | None, chunk_size: int) -> dict:
        def get_splits(tokens_row, mask_row, chunk_size, comma_id=None):
            tokens_splits = []
            mask_splits = []
            start = 0
            while start < len(tokens_row):
                if len(tokens_row) - start <= chunk_size:
                    tokens_splits.append(tokens_row[start:])
                    if mask_row is not None:
                        mask_splits.append(mask_row[start:])
                    break
                end = start + chunk_size
                if comma_id is not None:
                    found_comma = -1
                    for i in range(end - 1, start - 1, -1):
                        if tokens_row[i] == comma_id:
                            found_comma = i
                            break
                    if found_comma != -1:
                        end = found_comma + 1
                tokens_splits.append(tokens_row[start:end])
                if mask_row is not None:
                    mask_splits.append(mask_row[start:end])
                start = end
            return tokens_splits, mask_splits

        # split tokens into chunks of chunk_size tokens, and add BOS/EOS to each
        bos_token = tokens[0]
        eos_token = tokens[-1]

        # remove BOS and EOS
        tokens = tokens[1:-1]
        if tokens_attention_mask is not None:
            tokens_attention_mask = tokens_attention_mask[1:-1]

        comma_id = None
        if self.split_on_comma and self.tokenizer:
            comma_id = self.tokenizer.convert_tokens_to_ids(",")
            if isinstance(comma_id, list):
                comma_id = comma_id[0]

        input_id_chunks, attention_mask_chunks = get_splits(tokens, tokens_attention_mask, chunk_size, comma_id)

        if self.max_chunks is not None:
            input_id_chunks = input_id_chunks[:self.max_chunks]
            attention_mask_chunks = attention_mask_chunks[:self.max_chunks]

        # pad each chunk with EOS
        new_input_id_chunks = []
        new_attention_mask_chunks = []
        for i in range(len(input_id_chunks)):
            chunk = input_id_chunks[i]
            padding_len = chunk_size - len(chunk)

            if padding_len > 0:
                chunk = torch.cat([
                    chunk,
                    torch.full((padding_len,), eos_token, dtype=tokens.dtype, device=tokens.device)
                ])
                if tokens_attention_mask is not None:
                    mask_chunk = attention_mask_chunks[i]
                    mask_chunk = torch.cat([
                        mask_chunk,
                        torch.full((padding_len,), 0, dtype=tokens_attention_mask.dtype, device=tokens_attention_mask.device)
                    ])
                    new_attention_mask_chunks.append(mask_chunk)
            else:
                if tokens_attention_mask is not None:
                    new_attention_mask_chunks.append(attention_mask_chunks[i])

            new_input_id_chunks.append(chunk)

        input_id_chunks = new_input_id_chunks
        if tokens_attention_mask is not None:
            attention_mask_chunks = new_attention_mask_chunks
        else:
            attention_mask_chunks = [None] * len(input_id_chunks)

        # add BOS and EOS to each chunk
        input_id_chunks = [torch.cat([bos_token.unsqueeze(0), chunk, eos_token.unsqueeze(0)]) for chunk in input_id_chunks]
        if tokens_attention_mask is not None:
            attention_mask_chunks = [torch.cat([torch.ones(1, dtype=tokens_attention_mask.dtype, device=tokens_attention_mask.device), chunk, torch.ones(1, dtype=tokens_attention_mask.dtype, device=tokens_attention_mask.device)]) for chunk in attention_mask_chunks]

        hidden_states = []
        pooled_states = []

        for chunk_tokens, chunk_attention_mask in zip(input_id_chunks, attention_mask_chunks, strict=True):
            res = self._get_item_single(chunk_tokens, chunk_attention_mask)
            hidden_states.append(res[self.hidden_state_out_name])
            if self.pooled_out_name:
                pooled_states.append(res[self.pooled_out_name])

        hidden_state = torch.cat(hidden_states, dim=0)

        if self.pooled_out_name:
            handling = str(self.pooled_output_handling)
            if handling == 'FIRST':
                pooled_state = pooled_states[0]
            elif handling == 'AVERAGE':
                pooled_state = torch.mean(torch.stack(pooled_states), dim=0)
            else:
                pooled_state = pooled_states[0]
        else:
            pooled_state = None

        return {
            self.hidden_state_out_name: hidden_state,
            self.pooled_out_name: pooled_state,
        }
