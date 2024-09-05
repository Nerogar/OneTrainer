from torch import Tensor
import torch

from transformers import CLIPTextModel, CLIPTextModelWithProjection


def encode_clip(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor | None = None,
        default_layer: int = 0,
        layer_skip: int = 0,
        add_output: bool = True,
        text_encoder_output: Tensor | None = None,
        add_pooled_output: bool = False,
        pooled_text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
) -> tuple[Tensor, Tensor]:
    chunk_length = 75
    max_embeddings_multiples = 3

    if tokens is None or tokens.numel() == 0:
        return None, None

    chunks = [tokens[:, i:i + chunk_length] for i in range(0, tokens.shape[1], chunk_length)]
    chunk_embeddings = []
    pooled_outputs = []

    for i, chunk in enumerate(chunks):
        if chunk.numel() == 0:
            continue

        # Create attention mask (1 for non-masked, 0 for masked)
        chunk_attention_mask = torch.ones_like(chunk, dtype=torch.bool)

        # First, add BOS and EOS tokens
        bos_tokens = torch.full((chunk.shape[0], 1), text_encoder.config.bos_token_id, dtype=chunk.dtype, device=chunk.device)
        eos_tokens = torch.full((chunk.shape[0], 1), text_encoder.config.eos_token_id, dtype=chunk.dtype, device=chunk.device)
        chunk = torch.cat([bos_tokens, chunk, eos_tokens], dim=1)
        chunk_attention_mask = torch.cat([torch.zeros_like(bos_tokens, dtype=torch.bool) if i > 0 else torch.ones_like(bos_tokens, dtype=torch.bool),
                                    chunk_attention_mask, 
                                    torch.zeros_like(eos_tokens, dtype=torch.bool) if i < len(chunks) - 1 else torch.ones_like(eos_tokens, dtype=torch.bool)],
                                    dim=1)

        # Fill with padding
        if chunk.shape[1] < chunk_length + 2:  # +2 is for BOS and EOS
            padding = torch.full((chunk.shape[0], chunk_length + 2 - chunk.shape[1]), text_encoder.config.eos_token_id, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, padding], dim=1)
            chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros_like(padding, dtype=torch.bool)], dim=1)
        
        outputs = text_encoder(
            chunk,
            attention_mask=chunk_attention_mask if use_attention_mask else None,
            return_dict=True,
            output_hidden_states=True,
        )
        
        if add_output:
            embedding = outputs.hidden_states[default_layer - layer_skip]
            chunk_embeddings.append(embedding)
        if add_pooled_output:
            if hasattr(outputs, "text_embeds"):
                pooled_outputs.append(outputs.text_embeds)
            elif hasattr(outputs, "pooler_output"):
                pooled_outputs.append(outputs.pooler_output)

    if add_output:
        if chunk_embeddings and len(chunk_embeddings) > max_embeddings_multiples:
            chunk_embeddings = chunk_embeddings[:max_embeddings_multiples]
        text_encoder_output = torch.cat(chunk_embeddings, dim=1)
        if add_layer_norm:
            final_layer_norm = text_encoder.text_model.final_layer_norm
            text_encoder_output = final_layer_norm(text_encoder_output)
    else:
        text_encoder_output = None

    if add_pooled_output:
        if pooled_outputs and len(pooled_outputs) > max_embeddings_multiples:
            pooled_outputs = pooled_outputs[:max_embeddings_multiples]
        pooled_text_encoder_output = pooled_outputs[0] if pooled_outputs else None
    else:
        pooled_text_encoder_output = None

    return text_encoder_output, pooled_text_encoder_output