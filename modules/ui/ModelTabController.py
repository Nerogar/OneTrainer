

from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod


class ModelTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_presets(self) -> dict:
        cls = create.get_model_setup_class(self.train_config.model_type, self.train_config.training_method)
        return cls.LAYER_PRESETS if cls is not None else {"full": []}

    def get_output_formats(self) -> list[tuple[str, ModelFormat]]:
        if self.train_config.training_method == TrainingMethod.EMBEDDING:
            # embedding: a plain safetensors file of the learned vectors
            return [("Safetensors", ModelFormat.SAFETENSORS)]
        elif self.train_config.training_method == TrainingMethod.LORA:
            # LoRA output formats supported by this model (model_type.supported_lora_formats drops the
            # ones this model can't produce, e.g. LEGACY for HiDream/Sana/Wuerstchen v2).
            labels = {
                ModelFormat.DIFFUSERS_LORA: "Diffusers",
                ModelFormat.KOHYA_LORA: "Kohya",
                ModelFormat.ORIGINAL_LORA: "Original",
                ModelFormat.COMFY_LORA: "Comfy",
                ModelFormat.LEGACY_LORA: "Legacy",
            }
            return [(labels[fmt], fmt) for fmt in self.train_config.model_type.supported_lora_formats()]
        else:
            # full model output formats supported by this model (model_type.supported_full_model_formats drops the
            # ones it can't produce, e.g. no single-file for Sana / Wuerstchen v2, COMFY only for Z-Image).
            labels = {
                ModelFormat.DIFFUSERS: "Diffusers",
                ModelFormat.ORIGINAL_SINGLE_FILE: "Original (single file)",
                ModelFormat.ORIGINAL_TRANSFORMER: "Original (transformer only)",
                ModelFormat.COMFY_TRANSFORMER: "Comfy (transformer only)",
                ModelFormat.LEGACY_SAFETENSORS: "Legacy",
            }
            return [(labels[fmt], fmt) for fmt in self.train_config.model_type.supported_full_model_formats()]
