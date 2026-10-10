"""Logic behind the simple training UI.

The simple UI only exposes a handful of settings. Everything else comes from a
built-in preset in training_presets/, so a run is always "preset + the few
values the user changed". Nothing here depends on Qt.
"""
import json
import os
from contextlib import suppress
from dataclasses import asdict, dataclass, fields

from modules.ui.ModelTabController import ModelTabController
from modules.ui.TopBarController import TopBarController
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.path_util import write_json_atomic

PRESET_DIR = "training_presets"
SETTINGS_PATH = "training_user_settings/simple_ui.json"
CONCEPT_FILE = "training_concepts/simple_ui.json"
SAMPLE_FILE = "training_samples/simple_ui.json"
ADVANCED_STATE_FILE = "training_presets/#.json"

DEFAULT_MODEL_GROUP = "Anima"

# Simple-UI starting values that differ from the built-in preset, keyed by preset file name.
# anima LoRA: the preset's 3e-5 at rank 16 / alpha 1 is ~10x weaker than the model card's
# recommendation ("for a rank 32 LoRA, start with 2e-5", alpha = rank), so it needs ~30 epochs
# before the subject starts to show. Use the model card values as-is.
PRESET_OVERRIDES = {
    "#anima LoRA.json": {"learning_rate": 2e-5, "lora_rank": 32, "lora_alpha": 32.0},
}

# presets that need extra inputs (masks, embedding placeholders) the simple UI doesn't offer
_UNSUPPORTED_PRESET_WORDS = ("inpaint", "masked", "embedding")


@dataclass
class SimpleTrainSettings:
    preset_path: str = ""
    base_model_name: str = ""
    # a single-file checkpoint (e.g. a finetune from ComfyUI's diffusion_models folder) replacing the
    # base model's transformer; the base model still provides the text encoder, VAE and configs
    transformer_model_name: str = ""
    dataset_path: str = ""
    include_subdirectories: bool = False
    caption_source: str = "sample"  # "sample" = .txt file next to each image, "filename" = the image file name
    output_name: str = "my_lora"
    output_dir: str = "models"
    epochs: int = 100
    repeats: float = 1.0
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: float = 16.0
    resolution: str = "1024"
    batch_size: int = 1
    save_every_epochs: int = 5
    sample_prompt: str = ""
    sample_every_epochs: int = 5

    @staticmethod
    def load() -> "SimpleTrainSettings":
        settings = SimpleTrainSettings()
        with suppress(OSError, ValueError), open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            known = {f.name for f in fields(SimpleTrainSettings)}
            for key, value in data.items():
                if key in known:
                    setattr(settings, key, value)
        return settings

    def save(self):
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        write_json_atomic(SETTINGS_PATH, asdict(self))


@dataclass
class PresetInfo:
    name: str
    path: str
    training_method: TrainingMethod
    base_model_name: str
    learning_rate: float
    lora_rank: int
    lora_alpha: float
    resolution: str
    batch_size: int
    epochs: int
    output_extension: str  # "" when the output is a folder
    supports_transformer_override: bool


@dataclass
class DatasetInfo:
    image_count: int
    caption_count: int


class SimpleTrainController:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def load_preset_groups(self) -> dict[str, list[PresetInfo]]:
        """Built-in presets grouped by model folder, e.g. {"Anima": [LoRA, Finetune]}."""
        groups: dict[str, list[PresetInfo]] = {}
        for group_name, children in TopBarController(self.train_config).load_preset_tree(PRESET_DIR):
            if not isinstance(children, list):
                continue
            presets = []
            for name, path in children:
                if isinstance(path, list) or any(w in name.lower() for w in _UNSUPPORTED_PRESET_WORDS):
                    continue
                info = self._read_preset(name, path)
                if info is not None and info.training_method in (TrainingMethod.LORA, TrainingMethod.FINE_TUNE):
                    presets.append(info)
            if presets:
                # LoRA first: it is what most people want and it needs far less VRAM
                presets.sort(key=lambda p: (p.training_method != TrainingMethod.LORA, p.name.lower()))
                groups[group_name] = presets
        return groups

    @staticmethod
    def _read_preset(name: str, path: str) -> PresetInfo | None:
        try:
            with open(path, encoding="utf-8") as f:
                config = TrainConfig.default_values().from_dict(json.load(f), migrate=False)
        except (OSError, ValueError):
            return None
        for key, value in PRESET_OVERRIDES.get(os.path.basename(path), {}).items():
            setattr(config, key, value)
        return PresetInfo(
            name=name.lstrip("#").strip(),
            path=path,
            training_method=config.training_method,
            base_model_name=config.base_model_name,
            learning_rate=config.learning_rate,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            resolution=config.resolution,
            batch_size=config.batch_size,
            epochs=config.epochs,
            output_extension=config.output_model_format.file_extension(),
            supports_transformer_override=ModelTabController(config).supports_override_transformer(),
        )

    @staticmethod
    def scan_dataset(path: str, include_subdirectories: bool) -> DatasetInfo:
        image_count = caption_count = 0
        if not path or not os.path.isdir(path):
            return DatasetInfo(0, 0)
        walker = os.walk(path) if include_subdirectories else [(path, [], os.listdir(path))]
        for _, _, files in walker:
            names = set(files)
            for file in files:
                stem, ext = os.path.splitext(file)
                if path_util.is_supported_image_extension(ext) and not stem.endswith("-masklabel"):
                    image_count += 1
                    if f"{stem}.txt" in names:
                        caption_count += 1
        return DatasetInfo(image_count, caption_count)

    @staticmethod
    def estimate_steps(image_count: int, repeats: float, batch_size: int, epochs: int) -> int:
        steps_per_epoch = int(image_count * repeats) // max(batch_size, 1)
        return steps_per_epoch * epochs

    @staticmethod
    def output_path(settings: SimpleTrainSettings, config: TrainConfig) -> str:
        name = path_util.safe_filename(settings.output_name)
        return os.path.join(settings.output_dir, name + config.output_model_format.file_extension())

    def validate(self, settings: SimpleTrainSettings) -> list[str]:
        errors = []
        if not settings.preset_path or not os.path.isfile(settings.preset_path):
            errors.append("Please choose a model and training type.")
        base_model = settings.base_model_name.strip()
        if not base_model:
            errors.append("Please set the base model.")
        elif os.path.isfile(base_model):
            errors.append("The base model must be a Diffusers folder or a Hugging Face name, not a single file. "
                          "Put single .safetensors / .gguf files into \"Custom model file\".")
        transformer_model = settings.transformer_model_name.strip()
        if transformer_model and not os.path.isfile(transformer_model):
            errors.append("The custom model file does not exist.")
        if not settings.dataset_path or not os.path.isdir(settings.dataset_path):
            errors.append("The image folder does not exist.")
        elif self.scan_dataset(settings.dataset_path, settings.include_subdirectories).image_count == 0:
            errors.append("The image folder contains no images.")
        if not path_util.safe_filename(settings.output_name):
            errors.append("Please enter an output name.")
        if not settings.output_dir.strip():
            errors.append("Please choose an output folder.")
        if settings.epochs <= 0:
            errors.append("Epochs must be greater than 0.")
        if settings.learning_rate <= 0:
            errors.append("Learning rate must be greater than 0.")
        return errors

    def build_config(self, settings: SimpleTrainSettings) -> TrainConfig:
        """Load the preset into self.train_config and apply the simple settings on top of it."""
        loaded = TopBarController(self.train_config).load_config_from_file(settings.preset_path)
        if loaded is None:
            raise RuntimeError(f"Could not load preset {settings.preset_path}")

        config = self.train_config
        name = path_util.safe_filename(settings.output_name)

        config.base_model_name = settings.base_model_name.strip()
        if ModelTabController(config).supports_override_transformer():
            transformer_model = settings.transformer_model_name.strip()
            config.transformer.model_name = transformer_model
            if transformer_model.lower().endswith(".gguf"):
                config.transformer.weight_dtype = DataType.GGUF
        config.workspace_dir = path_util.canonical_join("workspace", name)
        config.cache_dir = path_util.canonical_join("workspace-cache", name)
        config.output_model_destination = self.output_path(settings, config)
        config.save_filename_prefix = f"{name}-"
        config.cloud.enabled = False

        config.epochs = settings.epochs
        config.learning_rate = settings.learning_rate
        config.resolution = settings.resolution.strip()
        config.batch_size = settings.batch_size
        if config.training_method == TrainingMethod.LORA:
            config.lora_rank = settings.lora_rank
            config.lora_alpha = settings.lora_alpha

        if settings.save_every_epochs > 0:
            config.save_every = settings.save_every_epochs
            config.save_every_unit = TimeUnit.EPOCH
        else:
            config.save_every = 0
            config.save_every_unit = TimeUnit.NEVER

        concept = ConceptConfig.default_values()
        concept.name = os.path.basename(os.path.normpath(settings.dataset_path))
        concept.path = settings.dataset_path
        concept.include_subdirectories = settings.include_subdirectories
        concept.balancing = settings.repeats
        concept.text.prompt_source = settings.caption_source

        samples = []
        if settings.sample_prompt.strip() and settings.sample_every_epochs > 0:
            sample = SampleConfig.default_values(config.model_type)
            sample.prompt = settings.sample_prompt.strip()
            samples.append(sample)
            config.sample_after = settings.sample_every_epochs
            config.sample_after_unit = TimeUnit.EPOCH
        else:
            config.sample_after_unit = TimeUnit.NEVER

        # written to files (not kept inline) so the advanced UI can open the same run
        config.concept_file_name = CONCEPT_FILE
        config.sample_definition_file_name = SAMPLE_FILE
        for path, items in ((CONCEPT_FILE, [concept]), (SAMPLE_FILE, samples)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_json_atomic(path, [item.to_dict() for item in items])
        config.concepts = None
        config.samples = None

        return config

    def export_to_advanced(self, settings: SimpleTrainSettings):
        """Make the advanced UI start with this run's settings on its next launch."""
        config = self.build_config(settings)
        write_json_atomic(ADVANCED_STATE_FILE, config.to_settings_dict(secrets=False))
