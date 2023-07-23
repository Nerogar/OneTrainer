import os
from typing import Optional, Callable

import torch
import rembg
from torch import Tensor, nn
from torchvision.transforms import functional, transforms
from tqdm.auto import tqdm

from modules.module.BaseImageMaskModel import BaseImageMaskModel, MaskSample
from modules.util import path_util

DEVICE = "cuda"


class RembgModel(BaseImageMaskModel):
    def __init__(self):
        self.rembg_session = rembg.new_session()

        self.smoothing_kernel_radius = None
        self.smoothing_kernel = self.__create_average_kernel(self.smoothing_kernel_radius)

        self.expand_kernel_radius = None
        self.expand_kernel = self.__create_average_kernel(self.expand_kernel_radius)

        self.image2Tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    @staticmethod
    def __create_average_kernel(kernel_radius: Optional[int]):
        if kernel_radius is None:
            return None

        kernel_size = kernel_radius * 2 + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding_mode='replicate',
                           padding=kernel_radius)
        kernel.weight.data = kernel_weights
        kernel.requires_grad_(False)
        kernel.to(DEVICE)
        return kernel

    @staticmethod
    def __get_sample_filenames(sample_dir: str) -> [str]:
        filenames = []
        for filename in os.listdir(sample_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in path_util.supported_image_extensions() and '-masklabel.png' not in filename:
                filenames.append(os.path.join(sample_dir, filename))

        return filenames

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

    def mask_image(
            self,
            filename: str,
            prompts: [str],
            mode: str = 'fill',
            threshold: float = 0.3,
            smooth_pixels: int = 5,
            expand_pixels: int = 10
    ):
        mask_sample = MaskSample(filename)

        if mode == 'fill' and mask_sample.get_mask_tensor() is not None:
            return

        if self.smoothing_kernel_radius != smooth_pixels:
            self.smoothing_kernel = self.__create_average_kernel(smooth_pixels)
            self.smoothing_kernel_radius = smooth_pixels

        if self.expand_kernel_radius != expand_pixels:
            self.expand_kernel = self.__create_average_kernel(expand_pixels)
            self.expand_kernel_radius = expand_pixels

        output = rembg.remove(mask_sample.get_image(), only_mask=True, session=self.rembg_session).convert("RGB")
        output = self.image2Tensor(output).to(DEVICE)

        predicted_mask = self.__process_mask(output, mask_sample.height, mask_sample.width, threshold)

        if mode == 'replace' or mode == 'fill':
            mask_sample.set_mask_tensor(predicted_mask)
        elif mode == 'add':
            mask_sample.add_mask_tensor(predicted_mask)
        elif mode == 'subtract':
            mask_sample.subtract_mask_tensor(predicted_mask)

        mask_sample.save_mask()

    def mask_images(
            self,
            filenames: list[str],
            prompts: list[str],
            mode: str = 'fill',
            threshold: float = 0.3,
            smooth_pixels: int = 5,
            expand_pixels: int = 10,
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
    ):
        if progress_callback is not None:
            progress_callback(0, len(filenames))
        for i, filename in enumerate(tqdm(filenames)):
            try:
                self.mask_image(filename, prompts, mode, threshold, smooth_pixels, expand_pixels)
            except Exception as e:
                if error_callback is not None:
                    error_callback(filename)
            if progress_callback is not None:
                progress_callback(i + 1, len(filenames))

    def mask_folder(
            self,
            sample_dir: str,
            prompts: list[str],
            mode: str = 'fill',
            threshold: float = 0.3,
            smooth_pixels: int = 5,
            expand_pixels: int = 10,
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
    ):
        filenames = self.__get_sample_filenames(sample_dir)
        self.mask_images(
            filenames=filenames,
            prompts=prompts,
            mode=mode,
            threshold=threshold,
            smooth_pixels=smooth_pixels,
            expand_pixels=expand_pixels,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )
