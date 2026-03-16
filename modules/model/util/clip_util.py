from typing import Any

from modules.util.enum.PooledOutputHandling import PooledOutputHandling

import torch
from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection


def encode_clip(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = -1,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
            or (add_pooled_output and pooled_text_encoder_output is None) \
            and text_encoder is not None:

        text_encoder_output = text_encoder(
            tokens,
            attention_mask=attention_mask if use_attention_mask else None,
            return_dict=True,
            output_hidden_states=True,
        )

        pooled_text_encoder_output = None
        if add_pooled_output:
            if hasattr(text_encoder_output, "text_embeds"):
                pooled_text_encoder_output = text_encoder_output.text_embeds
            if hasattr(text_encoder_output, "pooler_output"):
                pooled_text_encoder_output = text_encoder_output.pooler_output

        text_encoder_output = text_encoder_output.hidden_states[default_layer - layer_skip] if add_output else None

        if add_layer_norm and text_encoder_output is not None:
            final_layer_norm = text_encoder.text_model.final_layer_norm
            text_encoder_output = final_layer_norm(text_encoder_output)

    return text_encoder_output, pooled_text_encoder_output


def get_num_clip_chunks(tokens: Tensor, chunk_size: int, split_on_comma: bool = False, tokenizer: Any | None = None, max_chunks: int | None = None) -> int:
    content_tokens = tokens[0, 1:-1]
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            indices = (content_tokens == pad_token_id).nonzero()
            if len(indices) > 0:
                content_tokens = content_tokens[:indices[0].item()]

    if not split_on_comma or tokenizer is None:
        num_chunks = (len(content_tokens) + chunk_size - 1) // chunk_size if len(content_tokens) > 0 else 1
        return min(num_chunks, max_chunks) if max_chunks is not None else num_chunks

    comma_id = tokenizer.convert_tokens_to_ids(",")
    if isinstance(comma_id, list):
        comma_id = comma_id[0]

    num_chunks = 0
    start = 0
    while start < len(content_tokens):
        num_chunks += 1
        if len(content_tokens) - start <= chunk_size:
            break
        end = start + chunk_size
        if comma_id is not None:
            found_comma = -1
            for i in range(end - 1, start - 1, -1):
                if content_tokens[i] == comma_id:
                    found_comma = i
                    break
            if found_comma != -1:
                end = found_comma + 1
        start = end

    num_chunks = max(num_chunks, 1)
    return min(num_chunks, max_chunks) if max_chunks is not None else num_chunks


def encode_clip_chunked(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokenizer: Any | None = None,
        tokens: Tensor | None = None,
        default_layer: int = -1,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
        min_chunks: int = 1,
        pooled_output_handling: PooledOutputHandling = PooledOutputHandling.FIRST,
        split_on_comma: bool = False,
        max_chunks: int | None = None,
) -> tuple[Tensor, Tensor]:
    if (add_output and text_encoder_output is None) \
            or (add_pooled_output and pooled_text_encoder_output is None) \
            and text_encoder is not None:

        max_length = text_encoder.config.max_position_embeddings
        chunk_size = max_length - 2

        if tokens.shape[1] > max_length or min_chunks > 1:
            bos_token = tokens[:, 0:1]
            eos_token = tokens[:, -1:]
            content_tokens = tokens[:, 1:-1]
            batch_size = tokens.shape[0]

            content_mask = None
            if use_attention_mask and attention_mask is not None:
                content_mask = attention_mask[:, 1:-1]

            def get_splits(tokens_row, mask_row, chunk_size, comma_id=None):
                if tokenizer is not None:
                    pad_token_id = tokenizer.pad_token_id
                    if pad_token_id is not None:
                        indices = (tokens_row == pad_token_id).nonzero()
                        if len(indices) > 0:
                            tokens_row = tokens_row[:indices[0].item()]
                            if mask_row is not None:
                                mask_row = mask_row[:indices[0].item()]

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

            comma_token_id = None
            if split_on_comma and tokenizer is not None:
                comma_token_id = tokenizer.convert_tokens_to_ids(",")
                if isinstance(comma_token_id, list):
                    comma_token_id = comma_token_id[0]

            all_element_chunks = []
            all_element_mask_chunks = []
            for b in range(batch_size):
                t_splits, m_splits = get_splits(content_tokens[b], content_mask[b] if content_mask is not None else None, chunk_size, comma_token_id)
                if max_chunks is not None:
                    t_splits = t_splits[:max_chunks]
                    m_splits = m_splits[:max_chunks]
                all_element_chunks.append(t_splits)
                all_element_mask_chunks.append(m_splits)

            num_chunks = max(len(chunks) for chunks in all_element_chunks)
            num_chunks = max(num_chunks, min_chunks)
            if max_chunks is not None:
                num_chunks = min(num_chunks, max_chunks)

            # pad chunk lists with empty chunks
            for b in range(batch_size):
                while len(all_element_chunks[b]) < num_chunks:
                    all_element_chunks[b].append(torch.tensor([], device=tokens.device, dtype=tokens.dtype))
                    if content_mask is not None:
                        all_element_mask_chunks[b].append(torch.tensor([], device=tokens.device, dtype=content_mask.dtype))

            outputs = []
            pooled_outputs = []

            for i in range(num_chunks):
                chunk_list = []
                chunk_attention_mask_list = []

                for b in range(batch_size):
                    element_chunk = all_element_chunks[b][i]
                    element_chunk_len = len(element_chunk)
                    padding_len = chunk_size - element_chunk_len

                    # pad chunk with EOS
                    padding = eos_token[b].repeat(padding_len)
                    chunk_list.append(torch.cat([bos_token[b], element_chunk, padding, eos_token[b]]))

                    # attention mask
                    if content_mask is not None:
                        element_mask_chunk = all_element_mask_chunks[b][i]
                        m = torch.cat([
                            attention_mask[b, 0:1], # BOS
                            element_mask_chunk,
                            torch.zeros(padding_len, device=tokens.device, dtype=attention_mask.dtype),
                            attention_mask[b, -1:] # EOS
                        ])
                        chunk_attention_mask_list.append(m)

                chunk_tokens = torch.stack(chunk_list)
                chunk_attention_mask = torch.stack(chunk_attention_mask_list) if content_mask is not None else None

                chunk_output, chunk_pooled_output = encode_clip(
                    text_encoder=text_encoder,
                    tokens=chunk_tokens,
                    default_layer=default_layer,
                    layer_skip=layer_skip,
                    add_output=add_output,
                    add_pooled_output=add_pooled_output,
                    use_attention_mask=use_attention_mask,
                    attention_mask=chunk_attention_mask,
                    add_layer_norm=add_layer_norm,
                )

                outputs.append(chunk_output)
                pooled_outputs.append(chunk_pooled_output)

            text_encoder_output = torch.cat(outputs, dim=1) if add_output else None

            if add_pooled_output:
                if pooled_output_handling == PooledOutputHandling.FIRST:
                    pooled_text_encoder_output = pooled_outputs[0]
                elif pooled_output_handling == PooledOutputHandling.LAST:
                    pooled_text_encoder_output = pooled_outputs[-1]
                elif pooled_output_handling == PooledOutputHandling.AVERAGE:
                    pooled_text_encoder_output = torch.mean(torch.stack(pooled_outputs), dim=0)
                else:
                    pooled_text_encoder_output = pooled_outputs[-1]
            else:
                pooled_text_encoder_output = None
        else:
            text_encoder_output, pooled_text_encoder_output = encode_clip(
                text_encoder=text_encoder,
                tokens=tokens,
                default_layer=default_layer,
                layer_skip=layer_skip,
                add_output=add_output,
                text_encoder_output=text_encoder_output,
                add_pooled_output=add_pooled_output,
                pooled_text_encoder_output=pooled_text_encoder_output,
                use_attention_mask=use_attention_mask,
                attention_mask=attention_mask,
                add_layer_norm=add_layer_norm,
            )

    return text_encoder_output, pooled_text_encoder_output
