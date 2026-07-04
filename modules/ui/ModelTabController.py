

from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat


class ModelTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_presets(self) -> dict:
        cls = create.get_model_setup_class(self.train_config.model_type, self.train_config.training_method)
        return cls.LAYER_PRESETS if cls is not None else {"full": []}

    def supports_override_transformer(self) -> bool:
        model_type = self.train_config.model_type
        # The transformer override path exists only for these architectures; SD3, PixArt, Sana
        # and HiDream have a transformer but expose no override field.
        return (
            model_type.is_flux()
            or model_type.is_z_image()
            or model_type.is_ernie()
            or model_type.is_chroma()
            or model_type.is_qwen()
            or model_type.is_anima()
            or model_type.is_krea2()
            or model_type.is_hunyuan_video()
            or model_type.is_ideogram()
        )

    def get_output_formats(self) -> list[tuple[str, ModelFormat]]:
        labels = {
            ModelFormat.SAFETENSORS: "Safetensors",
            ModelFormat.DIFFUSERS_LORA: "Diffusers",
            ModelFormat.KOHYA_LORA: "Kohya",
            ModelFormat.ORIGINAL_LORA: "Original",
            ModelFormat.COMFY_LORA: "Comfy",
            ModelFormat.LEGACY_LORA: "Legacy",
            ModelFormat.DIFFUSERS: "Diffusers",
            ModelFormat.ORIGINAL_SINGLE_FILE: "Original (single file)",
            ModelFormat.ORIGINAL_TRANSFORMER: "Original (transformer only)",
            ModelFormat.COMFY_TRANSFORMER: "Comfy (transformer only)",
            ModelFormat.LEGACY_SAFETENSORS: "Legacy",
        }
        formats = self.train_config.model_type.supported_output_formats(self.train_config.training_method)
        return [(labels[fmt], fmt) for fmt in formats]
