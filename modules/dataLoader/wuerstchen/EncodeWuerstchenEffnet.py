from contextlib import nullcontext

from modules.model.WuerstchenModel import WuerstchenEfficientNetEncoder

from mgds.MGDS import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class EncodeWuerstchenEffnet(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            effnet_encoder: WuerstchenEfficientNetEncoder,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.effnet_encoder = effnet_encoder

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)

        if self.dtype:
            image = image.to(device=self.effnet_encoder.device, dtype=self.dtype)

        with self._all_contexts(self.autocast_contexts):
            image_embeddings = self.effnet_encoder(image.unsqueeze(0)).squeeze()

        return {
            self.out_name: image_embeddings
        }
