import os
from abc import ABCMeta

import torch
from torch import Tensor
from torchvision import transforms


class ModelSetupDebugMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupDebugMixin, self).__init__()

    def _save_image(self, image_tensor: Tensor, directory: str, name: str, step: int):
        path = os.path.join(directory, "step-" + str(step) + "-" + name + ".png")
        if not os.path.exists(directory):
            os.makedirs(directory)

        t = transforms.ToPILImage()

        image_tensor = image_tensor[0].unsqueeze(0)

        range_min = -1
        range_max = 1
        image_tensor = (image_tensor - range_min) / (range_max - range_min)

        image = t(image_tensor.squeeze())
        image.save(path)

    def _project_latent_to_image(self, latent_tensor: Tensor):
        generator = torch.Generator(device=latent_tensor.device)
        generator.manual_seed(42)
        channels = latent_tensor.shape[1]
        weight = torch.randn(
            size=(3, channels, 1, 1),
            generator=generator,
            device=latent_tensor.device,
            dtype=latent_tensor.dtype,
        )

        with torch.no_grad():
            result = torch.nn.functional.conv2d(latent_tensor, weight)
            result_min = result.min()
            result_max = result.max()
            result = (result - result_min) / (result_max - result_min)
            return result * 2 - 1
