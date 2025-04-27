from abc import ABCMeta
from collections.abc import Callable

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.util.config.TrainConfig import TrainEmbeddingConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection

import torch
from torch import Tensor

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    Gemma2Model,
    LlamaModel,
    T5EncoderModel,
)
from transformers.tokenization_utils import PreTrainedTokenizer, Trie


class ModelSetupEmbeddingMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _remove_added_embeddings_from_tokenizer(
            self,
            tokenizer: PreTrainedTokenizer,
    ):
        if tokenizer:
            added_tokens = list(filter(lambda item: not item[1].special, tokenizer._added_tokens_decoder.items()))
            for key, added_token in added_tokens:
                tokenizer._added_tokens_decoder.pop(key)
                tokenizer._added_tokens_encoder.pop(added_token.content)
            tokenizer.tokens_trie = Trie()
            tokenizer._update_trie()

    def _create_new_embedding(
            self,
            model: BaseModel,
            embedding_config: TrainEmbeddingConfig,
            tokenizer: PreTrainedTokenizer | None,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection | T5EncoderModel | Gemma2Model | LlamaModel | None,
            create_output_embedding_fn: Callable[[str], Tensor] | None = None,
    ) -> Tensor | None:
        if tokenizer is None or text_encoder is None:
            return None

        with torch.no_grad():
            initial_token_ids = tokenizer(
                embedding_config.initial_embedding_text,
                padding='do_not_pad',
                truncation=embedding_config.token_count is not None,
                add_special_tokens=False,
                max_length=embedding_config.token_count,
            ).input_ids
            pad_token_id = tokenizer(
                '*',
                padding='do_not_pad',
                truncation=True,
                add_special_tokens=False,
                max_length=1,
            ).input_ids[0]

            if embedding_config.token_count is not None:
                initial_token_ids += [pad_token_id] * (embedding_config.token_count - len(initial_token_ids))

            all_embeddings = text_encoder.get_input_embeddings().weight.data
            initial_embeddings = [all_embeddings[token_id] for token_id in initial_token_ids]
            vector = torch.stack(initial_embeddings)

            if embedding_config.is_output_embedding and create_output_embedding_fn is not None:
                token_count = len(initial_token_ids)

                with model.autocast_context:
                    vector = create_output_embedding_fn(
                        embedding_config.initial_embedding_text + token_count * '*',
                    )[:token_count]

        return vector

    def _add_embeddings_to_tokenizer(
            self,
            tokenizer: PreTrainedTokenizer,
            embeddings: list[BaseModelEmbedding],
    ) -> (Tensor, list[bool]):
        for embedding in embeddings:
            tokenizer.add_tokens(embedding.text_tokens)

    def _add_embedding_param_groups(
            self,
            embeddings: list[BaseModelEmbedding],
            parameter_group_collection: NamedParameterGroupCollection,
            embedding_learning_rate: float,
            prefix: str,
    ):
        for embedding in embeddings:
            parameter = embedding.output_vector if embedding.is_output_embedding else embedding.vector
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"{prefix}/{embedding.uuid}",
                display_name=f"{prefix}/{embedding.placeholder}",
                parameters=[parameter],
                learning_rate=embedding_learning_rate,
            ))

    def _normalize_output_embeddings(self, embeddings: list[BaseModelEmbedding]):
        with torch.no_grad():
            for embedding in embeddings:
                if embedding.is_output_embedding and embedding.output_vector.requires_grad:
                    std = embedding.output_vector.std(dim=1).mean()
                    embedding.output_vector.mul_(embedding.original_output_vector_std / std)
