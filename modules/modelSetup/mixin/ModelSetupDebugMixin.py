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
        image_tensor = image_tensor.clamp(0.0, 1.0)

        image = t(image_tensor.squeeze())
        image.save(path)

    def _project_latent_to_image(self, latent_tensor: Tensor):
        generator = torch.Generator(device=latent_tensor.device)
        generator.manual_seed(42)
        weight = torch.randn((3, 4, 1, 1), generator=generator, device=latent_tensor.device, dtype=latent_tensor.dtype)

        with torch.no_grad():
            return torch.nn.functional.conv2d(latent_tensor, weight) / 3.0
