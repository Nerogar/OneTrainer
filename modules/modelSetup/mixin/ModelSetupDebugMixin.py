import os
from abc import ABCMeta

import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


class ModelSetupDebugMixin(metaclass=ABCMeta):
    def __init__(self):
        super(ModelSetupDebugMixin, self).__init__()

    def _save_image(self, image_tensor: Tensor, directory: str, name: str, step: int, fromarray: bool = False):
        path = os.path.join(directory, "step-" + str(step) + "-" + name + ".png")
        if not os.path.exists(directory):
            os.makedirs(directory)

        if fromarray:
            image = Image.fromarray(image_tensor)
        else:
            t = transforms.ToPILImage()

            image_tensor = image_tensor[0].unsqueeze(0)

            range_min = -1
            range_max = 1
            image_tensor = (image_tensor - range_min) / (range_max - range_min)

            image = t(image_tensor.squeeze())

        image.save(path)

    def _save_text(self, text: str, directory: str, name: str, step: int):
        path = os.path.join(directory, "step-" + str(step) + "-" + name + ".txt")
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, "w") as f:
            f.write(text)

    def _decode_tokens(self, tokens:Tensor, tokenizer):
        return tokenizer.decode(
            token_ids=tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Decodes 4-channel latent to 3-channel RGB - technique appropriated from 
    # https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
    # Uses linear approximation based on first three channels of latent image (luminance, cyan/red, lime/purple)
    def _project_latent_to_image_sdxl(self, latent_tensor: Tensor):
        weights = (
            (60, -60, 25, -70),
            (60, -5, 15, -50),
            (60, 10, -5, -35)
        )

        weights_tensor = torch.t(torch.tensor(weights, dtype=latent_tensor.dtype).to(latent_tensor.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latent_tensor.dtype).to(latent_tensor.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latent_tensor, weights_tensor) \
                     + biases_tensor.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)

        return image_array

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
