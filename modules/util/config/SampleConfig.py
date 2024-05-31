from typing import Any

from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.config.BaseConfig import BaseConfig


class SampleConfig(BaseConfig):
    enabled: bool
    prompt: str
    negative_prompt: str
    height: int
    width: int
    seed: int
    random_seed: bool
    diffusion_steps: int
    cfg_scale: float
    noise_scheduler: NoiseScheduler

    sample_inpainting: bool
    base_image_path: str
    mask_image_path: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(SampleConfig, self).__init__(data)

    @staticmethod
    def default_values():
        data = []

        data.append(("enabled", True, bool, False))
        data.append(("prompt", "", str, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("height", 512, int, False))
        data.append(("width", 512, int, False))
        data.append(("seed", 42, int, False))
        data.append(("random_seed", False, bool, False))
        data.append(("diffusion_steps", 20, int, False))
        data.append(("cfg_scale", 7.0, float, False))
        data.append(("noise_scheduler", NoiseScheduler.DDIM, NoiseScheduler, False))

        data.append(("sample_inpainting", False, bool, False))
        data.append(("base_image_path", "", str, False))
        data.append(("mask_image_path", "", str, False))

        return SampleConfig(data)
