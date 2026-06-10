from torch import Tensor

from transformers import LlamaModel


def encode_llama(
        text_encoder: LlamaModel,
        tokens: Tensor | None = None,
        default_layer: int = -1,
        layer_skip: int = 0,
        text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        crop_start: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if text_encoder_output is None and text_encoder is not None:
        text_encoder_output = text_encoder(
            tokens,
            attention_mask=attention_mask if use_attention_mask else None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_state_output_index = default_layer - layer_skip
        text_encoder_output = text_encoder_output.hidden_states[hidden_state_output_index]

        if crop_start is not None:
            tokens = tokens[:, crop_start:]
            text_encoder_output = text_encoder_output[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:]

    return text_encoder_output, attention_mask, tokens
