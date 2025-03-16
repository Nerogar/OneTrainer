from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class VideoToolConfig(BaseConfig):
    input: str
    include_subdirectories: bool
    image_output: str
    video_output: str

    image_rate: float
    blur_threshold: float

    cut_split_enabled: bool
    maximum_length: float

    filter_object: str
    filter_behavior: str

    download_link: str
    download_output: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("input", "", str, False))
        data.append(("include_subdirectories", True, bool, False))
        data.append(("image_output", "", str, False))
        data.append(("video_output", "", str, False))

        data.append(("image_rate", 1.0, float, False))
        data.append(("blur_threshold", 0.2, float, False))

        data.append(("cut_split_enabled", True, bool, False))
        data.append(("maximum_length", 3.0, float, False))

        data.append(("filter_object", "NONE", str, False))
        data.append(("filter_behavior", "INCLUDE", str, False))

        data.append(("download_link", "", str, False))
        data.append(("download_output", "", str, False))

        return VideoToolConfig(data)
