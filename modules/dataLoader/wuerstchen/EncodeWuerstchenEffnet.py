from contextlib import nullcontext

import torch
from mgds.MGDS import PipelineModule

from modules.model.WuerstchenModel import WuerstchenEfficientNetEncoder


class EncodeWuerstchenEffnet(PipelineModule):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            effnet_encoder: WuerstchenEfficientNetEncoder,
            override_allow_mixed_precision: bool | None = None,
    ):
        super(EncodeWuerstchenEffnet, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.effnet_encoder = effnet_encoder
        self.override_allow_mixed_precision = override_allow_mixed_precision

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.in_name, index)

        image = image.to(device=image.device, dtype=self.pipeline.dtype)

        allow_mixed_precision = self.pipeline.allow_mixed_precision if self.override_allow_mixed_precision is None \
            else self.override_allow_mixed_precision

        image = image if allow_mixed_precision else image.to(self.effnet_encoder.dtype)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type, self.pipeline.dtype) if allow_mixed_precision \
                    else nullcontext():
                image_embeddings = self.effnet_encoder(image.unsqueeze(0)).squeeze()

        return {
            self.out_name: image_embeddings
        }