import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

from .sample import CaptionSample
from .utils import get_sample_filenames, is_empty_caption

logger = logging.getLogger(__name__)

class BaseImageCaptionModel(ABC):
    @abstractmethod
    def generate_caption(
        self,
        sample: CaptionSample,
        initial: str = "",
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = ""
    ) -> str:
        """
        Generate a caption for the given sample.
        """

    def caption_image(
        self,
        filename: str,
        initial: str = "",
        mode: str = "fill",
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = ""
    ) -> None:
        """
        Caption a single image based on the mode:
            - "replace": Always set a new caption.
            - "fill": Only set caption if the current one is empty.
            - "add": Append a new caption.
        """
        logger.info(f"Captioning image: {filename}")
        sample = CaptionSample(filename)
        if mode == "fill" and any(not is_empty_caption(c) for c in sample.captions):
            return
        caption = self.generate_caption(sample, initial, initial_caption, caption_prefix, caption_postfix)
        if mode in ("replace", "fill"):
            sample.set_captions([caption])
        elif mode == "add":
            sample.add_caption(caption)
        for c in sample.captions:
            logger.debug(f"Caption: {c}")
        sample.save_captions()

    def caption_images(
        self,
        filenames: list[str],
        initial: str = "",
        mode: str = "fill",
        progress_callback: Callable[[int, int], None] = None,
        error_callback: Callable[[str], None] = None,
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = "",
    ) -> None:
        logger.info(f"Caption images processing {len(filenames)} files")
        total = len(filenames)
        if progress_callback:
            progress_callback(0, total)
        for idx, filename in enumerate(filenames):
            try:
                self.caption_image(filename, initial, mode, initial_caption, caption_prefix, caption_postfix)
            except Exception as e:
                if error_callback:
                    error_callback(filename)
                logger.error(f"Error captioning {filename}: {e}", exc_info=True)
            if progress_callback:
                progress_callback(idx + 1, total)

    def caption_folder(
        self,
        sample_dir: str,
        initial: str = "",
        mode: str = "fill",
        progress_callback: Callable[[int, int], None] = None,
        error_callback: Callable[[str], None] = None,
        include_subdirectories: bool = False,
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = "",
    ) -> None:
        filenames = get_sample_filenames(sample_dir, include_subdirectories)
        logger.info(f"Caption folder processing {len(filenames)} files")
        self.caption_images(
            filenames, initial, mode, progress_callback, error_callback,
            initial_caption, caption_prefix, caption_postfix
        )
