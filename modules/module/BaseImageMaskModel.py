import os
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.util import path_util
from modules.util.image_util import load_image

import torch
from torch import Tensor
from torchvision.transforms import transforms

from PIL import Image
from tqdm import tqdm


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
            self.image = load_image(self.image_filename, 'RGB')
            self.height = self.image.height
            self.width = self.image.width

        return self.image

    def get_mask_tensor(self) -> Tensor:
        if self.mask_tensor is None and os.path.exists(self.mask_filename):
            mask = load_image(self.mask_filename, 'L')
            mask = self.image2Tensor(mask)
            mask = mask.to(self.device)
            self.mask_tensor = mask.unsqueeze(0)

        return self.mask_tensor

    def set_mask_tensor(self, mask_tensor: Tensor, alpha: float):
        self.mask_tensor = alpha * mask_tensor

    def add_mask_tensor(self, mask_tensor: Tensor, alpha: float, inverted: bool):
        mask = self.get_mask_tensor()

        if inverted:
            mask_tensor = 1.0 - mask_tensor

        if mask is None:
            mask = alpha * mask_tensor
        else:
            torch.add(mask, mask_tensor, alpha=alpha, out=mask)

        torch.clamp(mask, 0, 1, out=mask)

        self.mask_tensor = mask

    def subtract_mask_tensor(self, mask_tensor: Tensor, alpha: float, inverted: bool):
        mask = self.get_mask_tensor()

        if inverted:
            mask_tensor = 1.0 - mask_tensor

        if mask is None:
            mask = alpha * mask_tensor
        else:
            torch.subtract(mask, mask_tensor, alpha=alpha, out=mask)

        torch.clamp(mask, 0, 1, out=mask)

        self.mask_tensor = mask

    def blend_mask_tensor(self, mask_tensor: Tensor, alpha: float):
        mask = self.get_mask_tensor()
        if mask is None:
            mask = alpha * mask_tensor
        else:
            torch.add(mask, mask_tensor, alpha=alpha, out=mask)
            if alpha < 0.0:
                mask -= alpha
            mask /= 1 + alpha

        self.mask_tensor = mask

    def apply_mask(self, mode: str, mask_tensor: Tensor, alpha: float, inverted: bool):
        if mode in {'replace', 'fill'}:
            self.set_mask_tensor(mask_tensor, alpha)
        elif mode == 'add':
            self.add_mask_tensor(mask_tensor, alpha, inverted)
        elif mode == 'subtract':
            self.subtract_mask_tensor(mask_tensor, alpha, inverted)
        elif mode == 'blend':
            self.blend_mask_tensor(mask_tensor, alpha)
        else:
            raise ValueError("invalid mode")

    def save_mask(self):
        if self.mask_tensor is not None:
            mask = self.mask_tensor.cpu().squeeze()
            mask = self.tensor2Image(mask).convert('RGB')
            mask.save(self.mask_filename)


class BaseImageMaskModel(metaclass=ABCMeta):
    @staticmethod
    def __get_sample_filenames(sample_dir: str, include_subdirectories: bool = False) -> list[str]:
        sample_dir = Path(sample_dir)

        def __is_supported_image_extension(path: Path) -> bool:
            ext = path.suffix
            return path_util.is_supported_image_extension(ext) and '-masklabel.png' not in path.name

        recursive_prefix = "" if not include_subdirectories else "**/"
        return [str(p) for p in sample_dir.glob(f'{recursive_prefix}*') if __is_supported_image_extension(p)]

    @abstractmethod
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
                - blend: blends the new mask with the old one
            alpha (`float`): the blending factor to use for modes add, subtract and blend
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
        """

    def mask_images(
            self,
            filenames: list[str],
            prompts: list[str],
            mode: str = 'fill',
            alpha: float = 1.0,
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
                - blend: blends the new mask with the old one
            alpha (`float`): the blending factor to use for modes add, subtract and blend
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
        """

        if progress_callback is not None:
            progress_callback(0, len(filenames))
        for i, filename in enumerate(tqdm(filenames)):
            try:
                self.mask_image(filename, prompts, mode, alpha, threshold, smooth_pixels, expand_pixels)
            except Exception:
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
            alpha: float = 1.0,
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
            include_subdirectories: bool = False,
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
                - blend: blends the new mask with the old one
            alpha (`float`): the blending factor to use for modes add, subtract and blend
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
            include_subdirectories (`bool`): whether to include subdirectories when processing samples
        """

        filenames = self.__get_sample_filenames(sample_dir, include_subdirectories)
        self.mask_images(
            filenames=filenames,
            prompts=prompts,
            mode=mode,
            alpha=alpha,
            threshold=threshold,
            smooth_pixels=smooth_pixels,
            expand_pixels=expand_pixels,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )
