from abc import ABCMeta

from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
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
            tokenizer: PreTrainedTokenizer,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection | T5EncoderModel | Gemma2Model | LlamaModel,
            initial_embedding_text: str,
            token_count: int | None,
    ) -> Tensor:
        with torch.no_grad():
            initial_token_ids = tokenizer.encode(
                initial_embedding_text,
                add_special_tokens=False,
                max_length=token_count,
            )
            pad_token_id = tokenizer.encode(
                '*',
                add_special_tokens=False,
                max_length=1,
            )[0]

            if token_count is not None:
                initial_token_ids += [pad_token_id] * (token_count - len(initial_token_ids))

            all_embeddings = text_encoder.get_input_embeddings().weight.data
            initial_embeddings = [all_embeddings[token_id] for token_id in initial_token_ids]
            return torch.stack(initial_embeddings)

    def _add_embedding_to_tokenizer(
            self,
            tokenizer: PreTrainedTokenizer,
            embedding: list[str],
    ) -> (Tensor, list[bool]):
        tokenizer.add_tokens(embedding)

    def _add_embedding_param_groups(
            self,
            embedding_wrapper: AdditionalEmbeddingWrapper,
            parameter_group_collection: NamedParameterGroupCollection,
            embedding_learning_rate: float,
            prefix: str,
    ):
        for parameter, placeholder, name in zip(embedding_wrapper.additional_embeddings,
                                                embedding_wrapper.additional_embedding_placeholders,
                                                embedding_wrapper.additional_embedding_names, strict=True):
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"{prefix}/{name}",
                display_name=f"{prefix}/{placeholder}",
                parameters=[parameter],
                learning_rate=embedding_learning_rate,
            ))
