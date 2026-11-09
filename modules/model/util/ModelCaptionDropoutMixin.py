import inspect
from random import Random

from modules.util.config.TrainConfig import TrainConfig

import torch
from torch import Tensor


def _map_tensors(value, fn):
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, fn) for item in value)
    return value


def _replace_samples(value, unconditional, mask: Tensor):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        # MGDS can retain only the prefix of Llama hidden states used by HiDream.
        return type(value)(
            _replace_samples(item, empty, mask)
            for item, empty in zip(value, unconditional[:len(value)], strict=True)
        )
    # Cached encoder outputs can use a different precision from the current batch.
    unconditional = unconditional.to(device=value.device, dtype=value.dtype)
    return torch.where(mask.to(value.device).view(-1, *([1] * (value.ndim - 1))), unconditional, value)


class ModelCaptionDropoutMixin:
    supports_caption_dropout = False
    force_zeros_for_empty_prompt = False
    _recording_unconditional = False

    @torch.no_grad()
    def cache_unconditional(self, config: TrainConfig):
        """Capture empty tokens and conditioning after adapters/embeddings have been installed."""
        self._unconditional_inputs = {}
        self._unconditional_outputs = {}
        if not self.supports_caption_dropout or self.force_zeros_for_empty_prompt:
            return
        parts = self.model_type.text_encoder_parts()
        if not any(getattr(config, part).dropout_probability > 0 for part in parts):
            return

        kwargs = {'text': '', 'train_device': self.train_device}
        for name in inspect.signature(self.encode_text).parameters:
            if name.endswith(('_layer_skip', '_sequence_length')):
                config_name = name.replace('text_encoder_1_', 'text_encoder_')
                if hasattr(config, config_name):
                    kwargs[name] = getattr(config, config_name)
            elif name == 'apply_attention_mask':
                kwargs[name] = config.transformer.attention_mask

        self.materialize_only(*parts)
        self._recording_unconditional = True
        try:
            # The encoder-specific context may be a no-op and relies on this outer context.
            with self.autocast_context:
                self.encode_text(**kwargs)
        finally:
            self._recording_unconditional = False
            self.evict(*parts)

    def _prepare_caption_dropout(
            self, encoder: str, probability: float | None, rand: Random | None,
            cached_output, *inputs,
    ):
        if self._recording_unconditional:
            self._unconditional_inputs[encoder] = _map_tensors(inputs, lambda t: t.detach().cpu())
            return (None, *inputs)
        if probability is None or probability <= 0:
            return (None, *inputs)

        reference = cached_output if cached_output is not None else inputs[0]
        if reference is None:  # An optional encoder was excluded from the model.
            return (None, *inputs)
        if isinstance(reference, (tuple, list)):
            reference = reference[0]
        rand = rand if rand is not None else Random()
        dropped = [rand.random() < probability for _ in range(reference.shape[0])]
        if not any(dropped):
            return (None, *inputs)
        mask = torch.tensor(dropped, dtype=torch.bool, device=reference.device)
        if cached_output is not None or self.force_zeros_for_empty_prompt:
            return (mask, *inputs)

        # A live encoder must encode empty tokens in this forward pass, so its weights,
        # adapters and embeddings keep receiving the correct gradients as they change.
        empty_inputs = self._unconditional_inputs[encoder]
        return (None, *(_replace_samples(value, empty, mask)
                        for value, empty in zip(inputs, empty_inputs, strict=True)))

    def _apply_caption_dropout(self, encoder: str, mask: Tensor | None, *outputs):
        if self._recording_unconditional:
            self._unconditional_outputs[encoder] = _map_tensors(outputs, lambda t: t.detach().cpu())
            return outputs
        if mask is None:
            return outputs
        empty_outputs = (_map_tensors(outputs, torch.zeros_like) if self.force_zeros_for_empty_prompt
                         else self._unconditional_outputs[encoder])
        return tuple(_replace_samples(value, empty, mask)
                     for value, empty in zip(outputs, empty_outputs, strict=True))
