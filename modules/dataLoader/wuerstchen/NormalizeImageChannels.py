from typing import Tuple

import torchvision
from mgds.MGDS import PipelineModule


class NormalizeImageChannels(PipelineModule):
    def __init__(
            self,
            image_in_name: str, image_out_name: str,
            mean: Tuple[float, float, float],
            std: Tuple[float, float, float],
    ):
        super(NormalizeImageChannels, self).__init__()
        self.image_in_name = image_in_name
        self.image_out_name = image_out_name
        self.mean = mean
        self.std = std

        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def length(self) -> int:
        return self.get_previous_length(self.image_in_name)

    def get_inputs(self) -> list[str]:
        return [self.image_in_name]

    def get_outputs(self) -> list[str]:
        return [self.image_out_name]

    def get_item(self, index: int, requested_name: str = None) -> dict:
        image = self.get_previous_item(self.image_in_name, index)

        image = self.normalize(image)

        return {
            self.image_out_name: image
        }