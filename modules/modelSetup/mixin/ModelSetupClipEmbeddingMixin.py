from abc import ABCMeta

import torch
from torch import Tensor
from torch.nn import Embedding
from transformers import CLIPTokenizer, CLIPTextModel


class ModelSetupClipEmbeddingMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupClipEmbeddingMixin, self).__init__()

    def _create_new_embedding(
            self,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
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

    def _add_embeddings_to_clip(
            self,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            embeddings: list[(Tensor, list[str])],
    ) -> (Tensor, list[bool]):
        with torch.no_grad():
            untrainable_token_ids = [True] * text_encoder.get_input_embeddings().weight.data.shape[0]

            for embedding_vectors, text_tokens in embeddings:
                tokenizer.add_tokens(text_tokens)
                text_encoder.resize_token_embeddings(len(tokenizer))

                token_ids = tokenizer.encode(
                    text_tokens,
                    add_special_tokens=False,
                )

                for token_id, embedding_vector in zip(token_ids, embedding_vectors):
                    text_encoder.get_input_embeddings().weight.data[token_id] = embedding_vector
                    while len(untrainable_token_ids) <= token_id:
                        untrainable_token_ids.append(True)
                    untrainable_token_ids[token_id] = False

            original_token_embeds = text_encoder.get_input_embeddings().weight.data.clone()
            return original_token_embeds, untrainable_token_ids

    def _embeddigns_after_optimizer_step(
            self,
            embedding_layer: Embedding,
            original_token_embeds: Tensor,
            untrainable_token_embeds_mask: list[bool],
    ):
        # reset untrainable embeddings
        with torch.no_grad():
            embedding_layer.weight[untrainable_token_embeds_mask] = \
                original_token_embeds[untrainable_token_embeds_mask]