from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from modules.util.enum.DataType import DataType


class AdditionalEmbeddingWrapper(metaclass=ABCMeta):
    prefix: str
    orig_module: nn.Embedding
    additional_embeddings: list[Tensor]

    def __init__(self, orig_module: nn.Embedding, additional_embeddings: list[Tensor], dtype: DataType | None):
        super(AdditionalEmbeddingWrapper, self).__init__()

        self.orig_module = orig_module
        self.additional_embeddings = additional_embeddings

        self.is_applied = False
        self.orig_forward = self.orig_module.forward

    def forward(self, x, *args, **kwargs):
        weight = torch.cat([self.orig_module.weight] + self.additional_embeddings, dim=0)
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
