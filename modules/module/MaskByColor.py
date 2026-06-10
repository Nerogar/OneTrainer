
from modules.module.BaseImageMaskModel import BaseImageMaskModel, MaskSample

import torch
from torch import Tensor, nn
from torchvision.transforms import functional, transforms


class MaskByColor(BaseImageMaskModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        self.smoothing_kernel_radius = None
        self.smoothing_kernel = self.__create_average_kernel(self.smoothing_kernel_radius)

        self.expand_kernel_radius = None
        self.expand_kernel = self.__create_average_kernel(self.expand_kernel_radius)

        self.dot_kernel = self.__create_dot_kernel((1.0, 1.0, 1.0))

        self.image2Tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __create_average_kernel(self, kernel_radius: int | None):
        if kernel_radius is None:
            return None

        kernel_size = kernel_radius * 2 + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        kernel = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding_mode='replicate',
            padding=kernel_radius
        )
        kernel.weight.data = kernel_weights
        kernel.requires_grad_(False)
        kernel.to(self.device)
        return kernel

    def __create_dot_kernel(self, color: tuple[float, float, float]):
        kernel_weights = torch.tensor(color).view(1, 3, 1, 1)
        kernel = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=1, bias=False, padding_mode='replicate',
            padding=0
        )
        kernel.weight.data = kernel_weights
        kernel.requires_grad_(False)
        kernel.to(self.device, self.dtype)
        return kernel

    def __process_mask(self, mask: Tensor, target_height: int, target_width: int, threshold: float) -> Tensor:
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)

        mask = mask.mean(1).unsqueeze(1)
        if self.smoothing_kernel is not None:
            mask = self.smoothing_kernel(mask)
        mask = functional.resize(mask, [target_height, target_width])
        mask = (mask > threshold).float()
        if self.expand_kernel is not None:
            mask = self.expand_kernel(mask)
        mask = (mask > 0).float()

        return mask

    def __parse_color(self, color: str) -> tuple[float, float, float]:
        if len(color) == 7 and color.startswith('#'):
            color = color[1:]
        if len(color) == 8 and color.startswith('0x'):
            color = color[2:]
        if len(color) == 6:
            r = float(int(color[0:2], 16)) / 255.0
            g = float(int(color[2:4], 16)) / 255.0
            b = float(int(color[4:6], 16)) / 255.0

            return (r, g, b)

        return (0.0, 0.0, 0.0)

    def mask_image(
            self,
            filename: str,
            prompts: list[str],
            mode: str = 'fill',
            alpha: float = 1.0,
            threshold: float = 0.3,
            smooth_pixels: int = 5,
            expand_pixels: int = 10
    ):
        color = self.__parse_color(prompts[0] if prompts else "")

        mask_sample = MaskSample(filename, self.device)

        if mode == 'fill' and mask_sample.get_mask_tensor() is not None:
            return

        if self.smoothing_kernel_radius != smooth_pixels:
            self.smoothing_kernel = self.__create_average_kernel(smooth_pixels)
            self.smoothing_kernel_radius = smooth_pixels

        if self.expand_kernel_radius != expand_pixels:
            self.expand_kernel = self.__create_average_kernel(expand_pixels)
            self.expand_kernel_radius = expand_pixels

        image = mask_sample.get_image()
        image_tensor = self.image2Tensor(image) \
            .to(device=self.device, dtype=self.dtype) \
            .unsqueeze(0)

        color_tensor = torch.tensor(color, dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        similarity = image_tensor - color_tensor
        similarity = similarity * similarity
        similarity = self.dot_kernel(similarity)
        similarity = torch.sqrt(similarity)
        output = similarity.to(dtype=torch.float32)

        predicted_mask = self.__process_mask(output, mask_sample.height, mask_sample.width, threshold)
        mask_sample.apply_mask(mode, predicted_mask, alpha, True)

        mask_sample.save_mask()
