import threading

import torch
from torch import Tensor

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

CLIP_CHUNK_TOKENS = 77
CLIP_CONTENT_TOKENS = 75
SDXL_CAPTION_TOKEN_LIMITS = (75, 150, 225)
_TOKENIZER_LOCK_INIT = threading.Lock()


def validate_sdxl_caption_token_limit(token_limit: int) -> int:
    if type(token_limit) is not int or token_limit not in SDXL_CAPTION_TOKEN_LIMITS:
        raise ValueError(
            f"SDXL caption token limit must be 75, 150, or 225, got {token_limit}"
        )
    return token_limit


def sdxl_chunk_attention_mask(
        tokenizer_1: CLIPTokenizer,
        tokens_1: Tensor,
        tokenizer_2: CLIPTokenizer,
        tokens_2: Tensor,
        token_limit: int,
) -> Tensor:
    """Mask unused chunks shared by both text encoders."""
    token_limit = validate_sdxl_caption_token_limit(token_limit)
    max_chunks = token_limit // CLIP_CONTENT_TOKENS

    def chunk_counts(tokenizer: CLIPTokenizer, tokens: Tensor) -> Tensor:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        expected_length = max_chunks * CLIP_CHUNK_TOKENS
        if tokens.ndim != 2 or tokens.shape[-1] != expected_length:
            raise ValueError(
                f"Chunked SDXL tokens must have shape [batch, {expected_length}]"
            )
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("SDXL CLIP tokenizers must define EOS and PAD token IDs")

        chunks = tokens.reshape(tokens.shape[0], max_chunks, CLIP_CHUNK_TOKENS)
        # CLIP-G can emit its pad token as caption content.
        empty = chunks[:, :, 1].eq(tokenizer.eos_token_id) & chunks[:, :, 2:].eq(tokenizer.pad_token_id).all(dim=-1)
        present = ~empty
        present[:, 0] = True
        return present.sum(dim=-1)

    counts_1 = chunk_counts(tokenizer_1, tokens_1)
    counts_2 = chunk_counts(tokenizer_2, tokens_2).to(counts_1.device)
    # Match ComfyUI's truncation to the shared CLIP chunk count.
    aligned_counts = torch.minimum(counts_1, counts_2)
    chunk_indices = torch.arange(max_chunks, device=aligned_counts.device)
    active_chunks = chunk_indices.unsqueeze(0) < aligned_counts.unsqueeze(1)
    return active_chunks.repeat_interleave(CLIP_CHUNK_TOKENS, dim=-1).to(dtype=torch.long)


def sdxl_chunk_aware_unet_prediction(
        unet,
        sample: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        added_cond_kwargs: dict[str, Tensor],
        chunk_attention_mask: Tensor | None,
) -> Tensor:
    """Run each batch group at its active CLIP length and restore batch order."""
    if chunk_attention_mask is None:
        return unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    if chunk_attention_mask.ndim != 2 or chunk_attention_mask.shape[0] != sample.shape[0]:
        raise ValueError("SDXL chunk attention mask must have one row per UNet sample")
    if chunk_attention_mask.shape[1] != encoder_hidden_states.shape[1]:
        raise ValueError("SDXL chunk attention mask length must match encoder hidden states")
    if chunk_attention_mask.shape[1] % CLIP_CHUNK_TOKENS != 0:
        raise ValueError("SDXL chunk attention mask length must be a multiple of 77")

    active_chunk_counts = chunk_attention_mask.sum(dim=-1) // CLIP_CHUNK_TOKENS
    if torch.any(active_chunk_counts < 1):
        raise ValueError("Every SDXL prompt must retain at least its first CLIP chunk")

    if torch.all(active_chunk_counts == active_chunk_counts[0]):
        active_length = int(active_chunk_counts[0].item()) * CLIP_CHUNK_TOKENS
        return unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states[:, :active_length].contiguous(),
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    predictions = None
    batch_size = sample.shape[0]
    for chunk_count_tensor in torch.unique(active_chunk_counts, sorted=True):
        chunk_count = int(chunk_count_tensor.item())
        indices = torch.nonzero(active_chunk_counts == chunk_count_tensor, as_tuple=False).flatten()
        grouped_kwargs = {
            key: value.index_select(0, indices)
            if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size
            else value
            for key, value in added_cond_kwargs.items()
        }
        grouped_timestep = (
            timestep.index_select(0, indices)
            if isinstance(timestep, Tensor) and timestep.ndim > 0 and timestep.shape[0] == batch_size
            else timestep
        )
        grouped_prediction = unet(
            sample=sample.index_select(0, indices),
            timestep=grouped_timestep,
            encoder_hidden_states=encoder_hidden_states.index_select(0, indices)[:, :chunk_count * CLIP_CHUNK_TOKENS],
            added_cond_kwargs=grouped_kwargs,
        ).sample
        if predictions is None:
            predictions = torch.empty(
                (batch_size, *grouped_prediction.shape[1:]),
                device=grouped_prediction.device,
                dtype=grouped_prediction.dtype,
            )
        predictions.index_copy_(0, indices, grouped_prediction)

    assert predictions is not None
    return predictions


def _tokenizer_lock(tokenizer: CLIPTokenizer) -> threading.Lock:
    lock = getattr(tokenizer, "_mgds_tokenizer_lock", None)
    if lock is None:
        with _TOKENIZER_LOCK_INIT:
            lock = getattr(tokenizer, "_mgds_tokenizer_lock", None)
            if lock is None:
                lock = threading.Lock()
                tokenizer._mgds_tokenizer_lock = lock
    return lock


def tokenize_clip_chunks(
        tokenizer: CLIPTokenizer,
        text: str,
        token_limit: int,
) -> tuple[Tensor, Tensor]:
    token_limit = validate_sdxl_caption_token_limit(token_limit)

    with _tokenizer_lock(tokenizer):
        if token_limit == 75:
            tokenizer_output = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=CLIP_CHUNK_TOKENS,
                return_tensors="pt",
            )
            return tokenizer_output.input_ids[0], tokenizer_output.attention_mask[0]

        content_ids = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=token_limit,
            return_tensors="pt",
        ).input_ids[0].tolist()

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise ValueError("SDXL CLIP tokenizers must define BOS, EOS, and PAD token IDs")

    token_ids = []
    attention_mask = []
    for chunk_index in range(token_limit // CLIP_CONTENT_TOKENS):
        chunk_content = content_ids[
            chunk_index * CLIP_CONTENT_TOKENS:(chunk_index + 1) * CLIP_CONTENT_TOKENS
        ]
        chunk = [bos_token_id, *chunk_content, eos_token_id]
        chunk_mask = [1] * len(chunk)
        padding_length = CLIP_CHUNK_TOKENS - len(chunk)
        chunk.extend([pad_token_id] * padding_length)
        chunk_mask.extend([0] * padding_length)
        token_ids.extend(chunk)
        attention_mask.extend(chunk_mask)

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def encode_clip_chunks(
        text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
        tokens: Tensor,
        hidden_state_output_index: int,
        add_pooled_output: bool = False,
        chunk_attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[-1] == 0 or tokens.shape[-1] % CLIP_CHUNK_TOKENS != 0:
        raise ValueError(f"Chunked SDXL tokens must have shape [batch, N * {CLIP_CHUNK_TOKENS}]")

    batch_size, sequence_length = tokens.shape
    chunk_count = sequence_length // CLIP_CHUNK_TOKENS
    chunk_tokens = tokens.reshape(batch_size, chunk_count, CLIP_CHUNK_TOKENS)
    active_chunks = None
    if chunk_attention_mask is not None:
        if chunk_attention_mask.shape != tokens.shape:
            raise ValueError("SDXL chunk attention mask must match the token tensor shape")
        active_chunks = chunk_attention_mask.to(device=tokens.device).reshape(
            batch_size, chunk_count, CLIP_CHUNK_TOKENS
        ).any(dim=-1)
        chunk_indices = torch.nonzero(active_chunks, as_tuple=False)
        if chunk_indices.numel() == 0:
            raise ValueError("At least one SDXL CLIP chunk must be active")
        chunk_tokens = chunk_tokens[chunk_indices[:, 0], chunk_indices[:, 1]]
    else:
        chunk_tokens = chunk_tokens.reshape(batch_size * chunk_count, CLIP_CHUNK_TOKENS)
    encoder_output = text_encoder(
        chunk_tokens,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_state = encoder_output.hidden_states[hidden_state_output_index]
    if active_chunks is None:
        hidden_state = hidden_state.reshape(batch_size, sequence_length, hidden_state.shape[-1])
    else:
        hidden_dimension = hidden_state.shape[-1]
        hidden_state = hidden_state.new_zeros((batch_size * chunk_count, CLIP_CHUNK_TOKENS, hidden_dimension))
        flat_chunk_indices = chunk_indices[:, 0] * chunk_count + chunk_indices[:, 1]
        hidden_state = hidden_state.index_copy(0, flat_chunk_indices, encoder_output.hidden_states[
            hidden_state_output_index
        ])
        hidden_state = hidden_state.reshape(batch_size, sequence_length, hidden_dimension)

    pooled_state = None
    if add_pooled_output:
        pooled_state = getattr(encoder_output, "text_embeds", None)
        if pooled_state is None:
            pooled_state = getattr(encoder_output, "pooler_output", None)
        if pooled_state is None:
            raise ValueError("The SDXL pooled text encoder output is missing")
        if active_chunks is None:
            pooled_state = pooled_state.reshape(batch_size, chunk_count, -1)[:, 0]
        else:
            first_chunk_positions = torch.nonzero(chunk_indices[:, 1] == 0, as_tuple=False).flatten()
            first_chunk_batches = chunk_indices[first_chunk_positions, 0]
            pooled_state = pooled_state.new_zeros((batch_size, pooled_state.shape[-1])).index_copy(
                0, first_chunk_batches, pooled_state.index_select(0, first_chunk_positions)
            )

    return hidden_state, pooled_state
