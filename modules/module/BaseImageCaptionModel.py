import os
from abc import ABCMeta, abstractmethod
from typing import Callable

from PIL import Image
from tqdm import tqdm

from modules.util import path_util


class CaptionSample:
    def __init__(self, filename: str):
        self.image_filename = filename
        self.caption_filename = os.path.splitext(filename)[0] + ".txt"

        self.image = None
        self.caption = None

        self.height = 0
        self.width = 0

    def get_image(self) -> Image:
        if self.image is None:
            self.image = Image.open(self.image_filename).convert('RGB')
            self.height = self.image.height
            self.width = self.image.width

        return self.image

    def get_caption(self) -> str:
        if self.caption is None and os.path.exists(self.caption_filename):
            try:
                with open(self.caption_filename, "r") as f:
                    self.caption = f.readlines()[0]
            except:
                self.caption = ""

        return self.caption

    def set_caption(self, caption: str):
        self.caption = caption

    def save_caption(self):
        if self.caption is not None:
            try:
                with open(self.caption_filename, "w", encoding='utf-8') as f:
                    f.write(self.caption)
            except:
                pass


class BaseImageCaptionModel(metaclass=ABCMeta):
    @staticmethod
    def __get_sample_filenames(sample_dir: str, include_subfolders: bool = False) -> [str]:
        def __is_supported_image_extension(filename: str) -> bool:
            ext = os.path.splitext(filename)[1]
            return path_util.is_supported_image_extension(ext) and '-masklabel.png' not in filename

        filenames = []
        if include_subfolders:
            for root, _, files in os.walk(sample_dir):
                for filename in files:
                    if __is_supported_image_extension(filename):
                        filenames.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(sample_dir):
                if __is_supported_image_extension(filename):
                    filenames.append(os.path.join(sample_dir, filename))

        return filenames

    @abstractmethod
    def caption_image(
            self,
            filename: str,
            initial_caption: str = "",
            mode: str = 'fill',
    ):
        """
        Captions a sample

        Parameters:
            filename (`str`): a sample filename
            initial_caption (`str`): an initial caption. the generated caption will start with this string
            mode (`str`): can be one of
                - replace: creates new caption for all samples, even if a caption already exists
                - fill: creates new caption for all samples without a caption
        """
        pass

    def caption_images(
            self,
            filenames: [str],
            initial_caption: str = "",
            mode: str = 'fill',
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
    ):
        """
        Captions all samples in a list

        Parameters:
            filenames (`[str]`): a list of sample filenames
            initial_caption (`str`): an initial caption. the generated caption will start with this string
            mode (`str`): can be one of
                - replace: creates new caption for all samples, even if a caption already exists
                - fill: creates new caption for all samples without a caption
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
        """

        if progress_callback is not None:
            progress_callback(0, len(filenames))
        for i, filename in enumerate(tqdm(filenames)):
            try:
                self.caption_image(filename, initial_caption, mode)
            except Exception as e:
                if error_callback is not None:
                    error_callback(filename)
            if progress_callback is not None:
                progress_callback(i + 1, len(filenames))

    def caption_folder(
            self,
            sample_dir: str,
            initial_caption: str = "",
            mode: str = 'fill',
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
            include_subfolders: bool = False,
    ):
        """
        Captions all samples in a folder

        Parameters:
            sample_dir (`str`): directory where samples are located
            initial_caption (`str`): an initial caption. the generated caption will start with this string
            mode (`str`): can be one of
                - replace: creates new caption for all samples, even if a caption already exists
                - fill: creates new caption for all samples without a caption
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
        """

        filenames = self.__get_sample_filenames(sample_dir, include_subfolders)
        self.caption_images(
            filenames=filenames,
            initial_caption=initial_caption,
            mode=mode,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )
