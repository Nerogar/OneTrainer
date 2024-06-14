from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import CLIPTokenizer, T5Tokenizer


class AdditionalEmbeddingWrapper(metaclass=ABCMeta):
    tokenizer: CLIPTokenizer | T5Tokenizer
    prefix: str
    orig_module: nn.Embedding
    additional_embeddings: list[Tensor]
    additional_embedding_placeholders: list[str]
    additional_embedding_names: list[str]

    def __init__(
            self,
            tokenizer: CLIPTokenizer | T5Tokenizer,
            orig_module: nn.Embedding,
            additional_embeddings: list[Tensor],
            additional_embedding_placeholders: list[str],
            additional_embedding_names: list[str],
    ):
        super(AdditionalEmbeddingWrapper, self).__init__()

        self.orig_module = orig_module
        self.additional_embeddings = additional_embeddings
        self.additional_embedding_placeholders = additional_embedding_placeholders
        self.additional_embedding_names = additional_embedding_names
        self.original_token_count = len(tokenizer) - sum(x.shape[0] for x in self.additional_embeddings)

        self.is_applied = False
        self.orig_forward = self.orig_module.forward
        self.orig_median_norm = torch.norm(self.orig_module.weight, dim=1).median().item()

    def forward(self, x, *args, **kwargs):
        # ensure that the original weights only contain as many embeddings as the unmodified tokenizer can create
        orig_module_weight = self.orig_module.weight[0:self.original_token_count]
        weight = torch.cat([orig_module_weight] + self.additional_embeddings, dim=0)
        return F.embedding(
            input=x,
            weight=weight,
        )

    def parameters(self):
        return self.additional_embeddings

    def hook_to_module(self):
        if not self.is_applied:
            self.orig_module.forward = self.forward
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.is_applied = False

    def normalize_embeddings(self):
        with torch.no_grad():
            for additional_embedding in self.additional_embeddings:
                if additional_embedding.requires_grad:  # only normalize if the embedding is learned
                    copy = torch.nn.functional.normalize(additional_embedding)
                    copy.mul_(self.orig_median_norm)
                    additional_embedding.copy_(copy)
