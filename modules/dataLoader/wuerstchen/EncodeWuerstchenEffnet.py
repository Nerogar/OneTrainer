from contextlib import nullcontext

import torch
from mgds.MGDS import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from modules.model.WuerstchenModel import WuerstchenEfficientNetEncoder


class EncodeWuerstchenEffnet(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            effnet_encoder: WuerstchenEfficientNetEncoder,
            autocast_context: torch.autocast | None = None,
    ):
        super(EncodeWuerstchenEffnet, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.effnet_encoder = effnet_encoder

        self.autocast_context = nullcontext() if autocast_context is None else autocast_context
        self.autocast_enabled = isinstance(self.autocast_context, torch.autocast)

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image = self._get_previous_item(variation, self.in_name, index)

        image = image.to(device=self.effnet_encoder.device)

        if not self.autocast_enabled:
            image = image.to(self.effnet_encoder.dtype)

        with self.autocast_context:
            image_embeddings = self.effnet_encoder(image.unsqueeze(0)).squeeze()

        return {
            self.out_name: image_embeddings
        }