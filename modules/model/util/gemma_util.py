from torch import Tensor

from transformers import Gemma2Model


def encode_gemma(
        text_encoder: Gemma2Model,
        tokens: Tensor | None = None,
        default_layer: int = 0,
        layer_skip: int = 0,
        text_encoder_output: Tensor | None = None,
        use_attention_mask: bool = True,
        attention_mask: Tensor | None = None,
        add_layer_norm: bool = True,
) -> Tensor:
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
        if hidden_state_output_index != -1 and add_layer_norm:
            text_encoder_output = text_encoder.norm(text_encoder_output)

    return text_encoder_output
