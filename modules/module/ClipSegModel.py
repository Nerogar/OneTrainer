
from modules.module.BaseImageMaskModel import BaseImageMaskModel, MaskSample

import torch
from torch import Tensor, nn
from torchvision.transforms import functional

from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class ClipSegModel(BaseImageMaskModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.eval()
        self.model.to(self.device)

        self.smoothing_kernel_radius = None
        self.smoothing_kernel = self.__create_average_kernel(self.smoothing_kernel_radius)

        self.expand_kernel_radius = None
        self.expand_kernel = self.__create_average_kernel(self.expand_kernel_radius)

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

    def __process_mask(self, mask: Tensor, target_height: int, target_width: int, threshold: float) -> Tensor:
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)

        mask = torch.sigmoid(mask)
        mask = mask.mean(1).unsqueeze(1)
        if self.smoothing_kernel is not None:
            mask = self.smoothing_kernel(mask)
        mask = functional.resize(mask, [target_height, target_width])
        mask = (mask > threshold).float()
        if self.expand_kernel is not None:
            mask = self.expand_kernel(mask)
        mask = (mask > 0).float()

        return mask

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
        mask_sample = MaskSample(filename, self.device)

        if mode == 'fill' and mask_sample.get_mask_tensor() is not None:
            return

        if self.smoothing_kernel_radius != smooth_pixels:
            self.smoothing_kernel = self.__create_average_kernel(smooth_pixels)
            self.smoothing_kernel_radius = smooth_pixels

        if self.expand_kernel_radius != expand_pixels:
            self.expand_kernel = self.__create_average_kernel(expand_pixels)
            self.expand_kernel_radius = expand_pixels

        inputs = self.processor(text=prompts, images=[mask_sample.get_image()] * len(prompts), padding="max_length",
                                return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_mask = self.__process_mask(outputs.logits, mask_sample.height, mask_sample.width, threshold)
        mask_sample.apply_mask(mode, predicted_mask, alpha, False)

        mask_sample.save_mask()
