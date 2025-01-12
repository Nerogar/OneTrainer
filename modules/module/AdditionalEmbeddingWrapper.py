from abc import ABCMeta

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers import PreTrainedTokenizer


class AdditionalEmbeddingWrapper(metaclass=ABCMeta):
    tokenizer: PreTrainedTokenizer
    prefix: str
    orig_module: nn.Embedding
    additional_embeddings: list[Tensor]
    additional_embedding_placeholders: list[str]
    additional_embedding_names: list[str]

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            orig_module: nn.Embedding,
            additional_embeddings: list[Tensor],
            additional_embedding_placeholders: list[str],
            additional_embedding_names: list[str],
    ):
        super().__init__()

        self.orig_module = orig_module
        self.additional_embeddings = additional_embeddings
        self.additional_embedding_placeholders = additional_embedding_placeholders
        self.additional_embedding_names = additional_embedding_names
        self.original_token_count = len(tokenizer) - sum(x.shape[0] for x in self.additional_embeddings)

        self.is_applied = False
        self.orig_forward = self.orig_module.forward
        self.orig_median_norm = torch.norm(self.orig_module.weight, dim=1).median().item()

    def forward(self, x, *args, **kwargs):
        if torch.is_grad_enabled() \
                and len(self.additional_embeddings) > 0 \
                and all(not x.requires_grad for x in self.additional_embeddings):
            # This dummy tensor ensures that the following operation always has a grad_fn
            # if an additional embedding exists and is trained. This fixes an issue with
            # a combination of gradient checkpointing, layer offloading and additional
            # embedding training when stopping the additional embedding training early.
            dummy = torch.zeros(size=(1, self.orig_module.weight.shape[1]), dtype = self.orig_module.weight.dtype, device=self.orig_module.weight.device)
            dummy.requires_grad_(True)
            dummy = [dummy]
        else:
            dummy = []

        # ensure that the original weights only contain as many embeddings as the unmodified tokenizer can create
        orig_module_weight = self.orig_module.weight[0:self.original_token_count]
        weight = torch.cat([orig_module_weight] + self.additional_embeddings + dummy, dim=0)
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
