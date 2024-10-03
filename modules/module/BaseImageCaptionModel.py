import contextlib
import os
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.util import path_util

from PIL import Image
from tqdm import tqdm


class CaptionSample:
    def __init__(self, filename: str):
        self.image_filename = filename
        self.caption_filename = os.path.splitext(filename)[0] + ".txt"

        self.image = None
        self.captions = None

        self.height = 0
        self.width = 0

    def get_image(self) -> Image:
        if self.image is None:
            self.image = Image.open(self.image_filename).convert('RGB')
            self.height = self.image.height
            self.width = self.image.width

        return self.image

    def get_caption(self) -> str:
        if self.captions is None and os.path.exists(self.caption_filename):
            try:
                with open(self.caption_filename, "r") as f:
                    self.captions = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
            except Exception:
                self.captions = []

        return self.captions

    def set_caption(self, caption: str):
        self.captions = [caption]

    def add_caption(self, caption: str):
        self.captions.append(caption)

    def save_caption(self):
        if self.captions is not None:
            with contextlib.suppress(Exception), open(self.caption_filename, "w", encoding='utf-8') as f:
                f.write('\n'.join(self.captions))


class BaseImageCaptionModel(metaclass=ABCMeta):
    @staticmethod
    def __get_sample_filenames(sample_dir: str, include_subdirectories: bool = False) -> list[str]:
        sample_dir = Path(sample_dir)

        def __is_supported_image_extension(path: Path) -> bool:
            ext = path.suffix
            return path_util.is_supported_image_extension(ext) and '-masklabel.png' not in path.name

        recursive_prefix = "" if not include_subdirectories else "**/"
        return [str(p) for p in sample_dir.glob(f'{recursive_prefix}*') if __is_supported_image_extension(p)]

    @abstractmethod
    def generate_caption(
            self,
            caption_sample: CaptionSample,
            initial_caption: str = "",
    ) -> str:
        """
        Generates caption for a single CaptionSample

        Args:
            caption_sample (`CaptionSample`): the sample to caption
            initial_caption (`str`): the initial caption

        Returns: the generated caption
        """

    def caption_image(
            self,
            filename: str,
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
            mode: str = 'fill',
    ):
        """
        Captions a sample

        Parameters:
            filename (`str`): a sample filename
            initial_caption (`str`): an initial caption. the generated caption will start with this string
            caption_prefix (`str`): add this to the start of the generated caption (before initial caption)
            caption_postfix (`str`): add this to the end of the generated caption
            mode (`str`): can be one of
                - replace: creates a new caption for all samples, even if a caption already exists
                - fill: creates a new caption for all samples without a caption
                - add: creates a new caption for all samples, appending if a caption already exists
        """
        caption_sample = CaptionSample(filename)

        existing_caption = caption_sample.get_caption()
        if mode == 'fill' and existing_caption is not None and existing_caption != "":
            return

        predicted_caption = self.generate_caption(caption_sample, initial_caption, caption_prefix, caption_postfix)

        if mode == 'replace' or mode == 'fill':
            caption_sample.set_caption(predicted_caption)

        if mode == 'add':
            caption_sample.add_caption(predicted_caption)

        caption_sample.save_caption()

    def caption_images(
            self,
            filenames: list[str],
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
            mode: str = 'fill',
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
    ):
        """
        Captions all samples in a list

        Parameters:
            filenames (`[str]`): a list of sample filenames
            initial_caption (`str`): an initial caption. the generated caption will start with this string
            caption_prefix (`str`): add this to the start of the generated caption (before initial caption)
            caption_postfix (`str`): add this to the end of the generated caption
            mode (`str`): can be one of
                - replace: creates a new caption for all samples, even if a caption already exists
                - fill: creates a new caption for all samples without a caption
                - add: creates a new caption for all samples, appending if a caption already exists
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
        """

        if progress_callback is not None:
            progress_callback(0, len(filenames))
        for i, filename in enumerate(tqdm(filenames)):
            try:
                self.caption_image(filename, initial_caption, caption_prefix, caption_postfix, mode)
            except Exception:
                if error_callback is not None:
                    error_callback(filename)
            if progress_callback is not None:
                progress_callback(i + 1, len(filenames))

    def caption_folder(
            self,
            sample_dir: str,
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
            mode: str = 'fill',
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
            include_subdirectories: bool = False,
    ):
        """
        Captions all samples in a folder

        Parameters:
            sample_dir (`str`): directory where samples are located
            initial_caption (`str`): an initial caption. the generated caption will start with this string
            caption_prefix (`str`): add this to the start of the generated caption (before initial caption)
            caption_postfix (`str`): add this to the end of the generated caption
            mode (`str`): can be one of
                - replace: creates a new caption for all samples, even if a caption already exists
                - fill: creates a new caption for all samples without a caption
                - add: creates a new caption for all samples, appending if a caption already exists
            progress_callback (`Callable[[int, int], None]`): called after every processed image
            error_callback (`Callable[[str], None]`): called for every exception
            include_subdirectories (`bool`): whether to include subfolders when processing samples
        """

        filenames = self.__get_sample_filenames(sample_dir, include_subdirectories)
        self.caption_images(
            filenames=filenames,
            initial_caption=initial_caption,
            caption_prefix=caption_prefix,
            caption_postfix=caption_postfix,
            mode=mode,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )
