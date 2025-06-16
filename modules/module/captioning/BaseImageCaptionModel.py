import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Any,
)

from .captioning_util import get_sample_filenames, is_empty_caption
from .CaptionSample import CaptionSample

logger = logging.getLogger(__name__)

class BaseImageCaptionModel(ABC):
    @abstractmethod
    def generate_caption(
        self,
        sample: CaptionSample,
        prompt: str,
        generation_config: Any | None = None # Specific models will type hint this more narrowly
    ) -> str:
        """
        Generate a caption for the given sample using the provided prompt and generation configuration.
        """

    def caption_image(
        self,
        filename: str,
        prompt_text: str, # Renamed from 'initial'
        mode: str,
        ui_prefix: str,
        ui_postfix: str,
        generation_config: Any | None = None # To be passed to generate_caption
    ) -> None:
        """
        Caption a single image.
        Applies UI prefix/postfix *after* model generation and any filtering (like blacklist).
        """
        logger.info(f"Captioning image: {filename}")
        sample = CaptionSample(filename)
        if mode == "fill" and any(not is_empty_caption(c) for c in sample.captions):
            return

        # generate_caption (which might be wrapped for blacklisting) returns the core generated text.
        generated_part = self.generate_caption(
            sample=sample,
            prompt=prompt_text,
            generation_config=generation_config
        )

        # Apply UI prefix and postfix here
        final_caption = ui_prefix + generated_part + ui_postfix

        if mode in ("replace", "fill"):
            sample.set_captions([final_caption])
        elif mode == "add":
            sample.add_caption(final_caption)
        for c in sample.captions:
            logger.debug(f"Caption: {c}")
        sample.save_captions()

    def caption_images(
        self,
        filenames: list[str],
        prompt_text: str,
        mode: str,
        ui_prefix: str,
        ui_postfix: str,
        generation_config: Any | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        error_callback: Callable[[str], None] | None = None
    ) -> None:
        logger.info(f"Caption images processing {len(filenames)} files")
        total = len(filenames)
        if progress_callback:
            progress_callback(0, total)
        for idx, filename in enumerate(filenames):
            try:
                self.caption_image(
                    filename=filename,
                    prompt_text=prompt_text,
                    mode=mode,
                    ui_prefix=ui_prefix,
                    ui_postfix=ui_postfix,
                    generation_config=generation_config
                )
            except Exception as e:
                if error_callback:
                    error_callback(filename)
                logger.error(f"Error captioning {filename}: {e}", exc_info=True)
            if progress_callback:
                progress_callback(idx + 1, total)

    def caption_folder(
        self,
        sample_dir: str,
        prompt_text: str,
        mode: str,
        ui_prefix: str,
        ui_postfix: str,
        generation_config: Any | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
        include_subdirectories: bool = False
    ) -> None:
        filenames = get_sample_filenames(sample_dir, include_subdirectories)
        logger.info(f"Caption folder processing {len(filenames)} files")
        self.caption_images(
            filenames=filenames,
            prompt_text=prompt_text,
            mode=mode,
            ui_prefix=ui_prefix,
            ui_postfix=ui_postfix,
            generation_config=generation_config,
            progress_callback=progress_callback,
            error_callback=error_callback
        )
