import os
import webbrowser

from modules.util import path_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.path_util import write_json_atomic


class TopBarController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_model_types(self) -> list[tuple[str, ModelType]]:
        return [  #TODO simplify
            ("SD1.5", ModelType.STABLE_DIFFUSION_15),
            ("SD1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("SD2.0", ModelType.STABLE_DIFFUSION_20),
            ("SD2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("SD2.1", ModelType.STABLE_DIFFUSION_21),
            ("SD3", ModelType.STABLE_DIFFUSION_3),
            ("SD3.5", ModelType.STABLE_DIFFUSION_35),
            ("SDXL", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("SDXL Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ("Flux Dev.1", ModelType.FLUX_DEV_1),
            ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
            ("Flux 2 [Dev, Klein]", ModelType.FLUX_2),
            ("Sana", ModelType.SANA),
            ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
            ("HiDream Full", ModelType.HI_DREAM_FULL),
            ("Chroma1", ModelType.CHROMA_1),
            ("QwenImage", ModelType.QWEN),
            ("Krea 2", ModelType.KREA_2),
            ("Z-Image", ModelType.Z_IMAGE),
            ("Ernie Image", ModelType.ERNIE),
        ]

    def get_training_methods(self, model_type: ModelType) -> list[tuple[str, TrainingMethod]]:
        #TODO simplify
        if model_type.is_stable_diffusion():
            return [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
                ("Fine Tune VAE", TrainingMethod.FINE_TUNE_VAE),
            ]
        elif model_type.is_stable_diffusion_3() \
                or model_type.is_stable_diffusion_xl() \
                or model_type.is_wuerstchen() \
                or model_type.is_pixart() \
                or model_type.is_flux_1() \
                or model_type.is_sana() \
                or model_type.is_hunyuan_video() \
                or model_type.is_hi_dream() \
                or model_type.is_chroma():
            return [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
            ]
        elif model_type.is_qwen() \
                or model_type.is_krea2() \
                or model_type.is_z_image() \
                or model_type.is_flux_2() \
                or model_type.is_ernie():
            return [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
            ]
        return []

    def load_available_config_names(self, dir: str) -> list[tuple[str, str]]:
        configs = [("", path_util.canonical_join(dir, "#.json"))]
        if os.path.isdir(dir):
            for path in os.listdir(dir):
                if path != "#.json":
                    path = path_util.canonical_join(dir, path)
                    if path.endswith(".json") and os.path.isfile(path):
                        name = os.path.basename(path)
                        name = os.path.splitext(name)[0]
                        configs.append((name, path))
            configs.sort()
        return configs

    def save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))
        return path

    def save_secrets(self, path) -> str:
        write_json_atomic(path, self.train_config.secrets.to_dict())
        return path

    def open_wiki(self):
        webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki", new=0, autoraise=False)

    def save_default(self):
        self.save_to_file("#")
        self.save_secrets("secrets.json")
