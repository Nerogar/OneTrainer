from __future__ import annotations

import os
from pathlib import Path

import pillow_jxl  # noqa: F401  # Needed for plugin registration
from PIL import Image


class CaptionSample:
    def __init__(self, filename: str) -> None:
        self.image_filename: str = filename
        # Save caption file next to the image
        self.caption_filename: str = str(Path(filename).with_suffix(".txt"))
        self._image: Image.Image | None = None
        self._captions: list[str] | None = None

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            self._image = Image.open(self.image_filename).convert("RGB")
        return self._image

    def get_image(self) -> Image.Image:
        return self.image

    @property
    def captions(self) -> list[str]:
        if self._captions is None and os.path.exists(self.caption_filename):
            try:
                with open(self.caption_filename, "r", encoding="utf-8") as f:
                    self._captions = [line.strip() for line in f if line.strip()]
            except Exception:
                self._captions = []
        return self._captions or []

    def set_captions(self, captions: list[str]) -> None:
        self._captions = captions

    def add_caption(self, caption: str) -> None:
        if self._captions is None:
            self._captions = []
        self._captions.append(caption)

    def save_captions(self) -> None:
        if self._captions is not None:
            try:
                with open(self.caption_filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(self._captions))
            except Exception:
                pass
