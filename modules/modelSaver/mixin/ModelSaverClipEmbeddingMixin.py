from abc import ABCMeta

import torch
from torch import Tensor
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection


class ModelSaverClipEmbeddingMixin(metaclass=ABCMeta):

    def _get_embedding_vector(
            self,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection,
            text_tokens: list[str],
    ) -> Tensor:
        with torch.no_grad():
            token_ids = tokenizer.encode(
                text_tokens,
                add_special_tokens=False,
            )

            all_embeddings = text_encoder.get_input_embeddings().weight.data
            initial_embeddings = [all_embeddings[token_id] for token_id in token_ids]
            return torch.stack(initial_embeddings)
