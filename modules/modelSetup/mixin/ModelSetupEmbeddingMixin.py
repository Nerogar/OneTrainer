from abc import ABCMeta

import torch
from torch import Tensor
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection, T5Tokenizer, T5EncoderModel
from transformers.tokenization_utils import Trie


class ModelSetupEmbeddingMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupEmbeddingMixin, self).__init__()

    def _remove_added_embeddings_from_tokenizer(
            self,
            tokenizer: CLIPTokenizer | T5Tokenizer,
    ):
        added_tokens = list(filter(lambda item: not item[1].special, tokenizer._added_tokens_decoder.items()))
        for key, added_token in added_tokens:
            tokenizer._added_tokens_decoder.pop(key)
            tokenizer._added_tokens_encoder.pop(added_token.content)
        tokenizer.tokens_trie = Trie()
        tokenizer._update_trie()

    def _create_new_embedding(
            self,
            tokenizer: CLIPTokenizer | T5Tokenizer,
            text_encoder: CLIPTextModel | CLIPTextModelWithProjection | T5EncoderModel,
            initial_embedding_text: str,
            token_count: int,
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
                max_length=token_count,
            )[0]
            initial_token_ids += [pad_token_id] * (token_count - len(initial_token_ids))

            all_embeddings = text_encoder.get_input_embeddings().weight.data
            initial_embeddings = [all_embeddings[token_id] for token_id in initial_token_ids]
            return torch.stack(initial_embeddings)

    def _add_embedding_to_tokenizer(
            self,
            tokenizer: CLIPTokenizer | T5Tokenizer,
            embedding: list[str],
    ) -> (Tensor, list[bool]):
        tokenizer.add_tokens(embedding)
