from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.enum.NoiseScheduler import NoiseScheduler


class SampleConfig(BaseConfig):
    enabled: bool
    prompt: str
    negative_prompt: str
    height: int
    width: int
    frames: int
    length: float
    seed: int
    random_seed: bool
    diffusion_steps: int
    cfg_scale: float
    noise_scheduler: NoiseScheduler

    text_encoder_1_layer_skip: int
    text_encoder_2_layer_skip: int
    text_encoder_2_sequence_length: int | None
    text_encoder_3_layer_skip: int
    text_encoder_4_layer_skip: int
    transformer_attention_mask: bool
    force_last_timestep: bool

    sample_inpainting: bool
    base_image_path: str
    mask_image_path: str

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    def from_train_config(self, train_config):
        self.text_encoder_1_layer_skip = train_config.text_encoder_layer_skip
        self.text_encoder_2_layer_skip = train_config.text_encoder_2_layer_skip
        self.text_encoder_2_sequence_length = train_config.text_encoder_2_sequence_length
        self.text_encoder_3_layer_skip = train_config.text_encoder_3_layer_skip
        self.text_encoder_4_layer_skip = train_config.text_encoder_4_layer_skip
        self.transformer_attention_mask = train_config.transformer.attention_mask
        self.force_last_timestep = train_config.rescale_noise_scheduler_to_zero_terminal_snr

    @staticmethod
    def default_values():
        data = []

        data.append(("enabled", True, bool, False))
        data.append(("prompt", "", str, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("height", 512, int, False))
        data.append(("width", 512, int, False))
        data.append(("frames", 1, int, False))
        data.append(("length", 10.0, float, False))
        data.append(("seed", 42, int, False))
        data.append(("random_seed", False, bool, False))
        data.append(("diffusion_steps", 20, int, False))
        data.append(("cfg_scale", 7.0, float, False))
        data.append(("noise_scheduler", NoiseScheduler.DDIM, NoiseScheduler, False))

        data.append(("text_encoder_1_layer_skip", 0, int, False))
        data.append(("text_encoder_2_layer_skip", 0, int, False))
        data.append(("text_encoder_2_sequence_length", None, int, True))
        data.append(("text_encoder_3_layer_skip", 0, int, False))
        data.append(("text_encoder_4_layer_skip", 0, int, False))
        data.append(("transformer_attention_mask", False, bool, False))
        data.append(("force_last_timestep", False, bool, False))

        data.append(("sample_inpainting", False, bool, False))
        data.append(("base_image_path", "", str, False))
        data.append(("mask_image_path", "", str, False))

        return SampleConfig(data)
