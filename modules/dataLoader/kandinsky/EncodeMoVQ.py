from contextlib import nullcontext

import torch
from diffusers import VQModel
from mgds.MGDS import PipelineModule


class EncodeMoVQ(PipelineModule):
    def __init__(
            self,
            in_name: str,
            out_name: str,
            movq: VQModel,
    ):
        super(EncodeMoVQ, self).__init__()
        self.in_name = in_name
        self.out_name = out_name
        self.movq = movq

    def length(self) -> int:
        return self.get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.in_name, index)

        image = image.to(device=image.device, dtype=self.pipeline.dtype)

        with torch.no_grad():
            with torch.autocast(self.pipeline.device.type) if self.pipeline.allow_mixed_precision else nullcontext():
                latent_image = self.movq.encode(image.unsqueeze(0)).latents

        latent_image = latent_image.squeeze()

        return {
            self.out_name: latent_image,
        }
