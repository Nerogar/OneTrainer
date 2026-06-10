from abc import ABCMeta

from modules.model.BaseModel import BaseModelEmbedding

import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedTokenizer


class AdditionalEmbeddingWrapper(metaclass=ABCMeta):
    tokenizer: PreTrainedTokenizer
    prefix: str
    orig_module: nn.Embedding
    embeddings: list[BaseModelEmbedding]

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            orig_module: nn.Embedding,
            embeddings: list[BaseModelEmbedding],
    ):
        super().__init__()

        self.orig_module = orig_module
        self.embeddings = embeddings
        self.original_token_count = len(tokenizer) - sum(embedding.token_count for embedding in self.embeddings)
        self.pad_vector = None

        self.is_applied = False
        self.orig_forward = self.orig_module.forward
        self.orig_median_norm = torch.norm(self.orig_module.weight, dim=1).median().item()

    def forward(self, x, *args, **kwargs):
        # ensure that the original weights only contain as many embeddings as the unmodified tokenizer can create
        orig_module_weight = self.orig_module.weight[0:self.original_token_count]

        # if the original weights don't contain enough vectors, pad with zero vectors
        pad_vector = []
        if orig_module_weight.shape[0] < self.original_token_count:
            pad_count = self.original_token_count - orig_module_weight.shape[0]
            pad_vector = [torch.zeros(size=(pad_count, self.orig_module.weight.shape[1]),
                                      dtype=self.orig_module.weight.dtype, device=self.orig_module.weight.device,
                                      requires_grad=False)]

        weight = torch.cat(
            [orig_module_weight] \
            + pad_vector \
            + [embedding.vector for embedding in self.embeddings],
            dim=0
        )

        return F.embedding(
            input=x,
            weight=weight,
        )

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
            for embedding in self.embeddings:
                if embedding.vector.requires_grad:  # only normalize if the embedding is learned
                    copy = torch.nn.functional.normalize(embedding.vector)
                    copy.mul_(self.orig_median_norm)
                    embedding.vector.copy_(copy)
