import os
from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms


class MaskSample:
    def __init__(self, filename: str, device: torch.device):
        self.image_filename = filename
        self.mask_filename = os.path.splitext(filename)[0] + "-masklabel.png"
        self.device = device

        self.image = None
        self.mask_tensor = None

        self.height = 0
        self.width = 0

        self.image2Tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.tensor2Image = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def get_image(self) -> Image:
        if self.image is None:
            self.image = Image.open(self.image_filename).convert('RGB')
            self.height = self.image.height
            self.width = self.image.width

        return self.image

    def get_mask_tensor(self) -> Tensor:
        if self.mask_tensor is None and os.path.exists(self.mask_filename):
            mask = Image.open(self.mask_filename).convert('L')
            mask = self.image2Tensor(mask)
            mask = mask.to(self.device)
            self.mask_tensor = mask.unsqueeze(0)

        return self.mask_tensor

    def set_mask_tensor(self, mask_tensor: Tensor):
        self.mask_tensor = mask_tensor

    def add_mask_tensor(self, mask_tensor: Tensor):
        mask = self.get_mask_tensor()
        if mask is None:
            mask = mask_tensor
        else:
            mask += mask_tensor
        mask = torch.clamp(mask, 0, 1)

        self.mask_tensor = mask

    def subtract_mask_tensor(self, mask_tensor: Tensor):
        mask = self.get_mask_tensor()
        if mask is None:
            mask = mask_tensor
        else:
            mask -= mask_tensor
        mask = torch.clamp(mask, 0, 1)

        self.mask_tensor = mask

    def save_mask(self):
        if self.mask_tensor is not None:
            mask = self.mask_tensor.cpu().squeeze()
            mask = self.tensor2Image(mask).convert('RGB')
            mask.save(self.mask_filename)


class BaseImageMaskModel(metaclass=ABCMeta):
    def mask_image(
            self,
            filename: str,
            prompts: [str],
            mode: str = 'fill',
            threshold: float = 0.3,
            smooth_pixels: int = 5,
            expand_pixels: int = 10
    ):
        """
        Masks a sample

        Parameters:
            filename (`str`): a sample filename
            prompts (`[str]`): a list of prompts used to create a mask
            mode (`str`): can be one of
                - replace: creates new masks for all samples, even if a mask already exists
                - fill: creates new masks for all samples without a mask
                - add: adds the new region to existing masks
                - subtract: subtracts the new region from existing masks
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
        """
        pass

    @abstractmethod
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
        """
        Masks all samples in a list

        Parameters:
            filenames (`[str]`): a list of sample filenames
            prompts (`[str]`): a list of prompts used to create a mask
            mode (`str`): can be one of
                - replace: creates new masks for all samples, even if a mask already exists
                - fill: creates new masks for all samples without a mask
                - add: adds the new region to existing masks
                - subtract: subtracts the new region from existing masks
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
        """
        pass

    @abstractmethod
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
        """
        Masks all samples in a folder

        Parameters:
            sample_dir (`str`): directory where samples are located
            prompts (`[str]`): a list of prompts used to create a mask
            mode (`str`): can be one of
                - replace: creates new masks for all samples, even if a mask already exists
                - fill: creates new masks for all samples without a mask
                - add: adds the new region to existing masks
                - subtract: subtracts the new region from existing masks
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
        """
        pass
