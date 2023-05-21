import argparse
from enum import Enum

from modules.util.args.arg_type_util import *
from modules.util.enum.DataType import DataType
from modules.util.enum.LossFunction import LossFunction
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod


class TrainArgs:
    training_method: TrainingMethod
    debug_mode: bool
    debug_dir: str
    workspace_dir: str
    cache_dir: str
    tensorboard: bool

    # model settings
    model_type: ModelType
    base_model_name: str
    extra_model_name: str
    output_dtype: DataType
    output_model_format: ModelFormat
    output_model_destination: str

    # data settings
    concept_file_name: str
    circular_mask_generation: bool
    random_rotate_and_crop: bool
    aspect_ratio_bucketing: bool
    latent_caching: bool
    latent_caching_epochs: int

    # training settings
    optimizer: Optimizer
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    train_text_encoder: bool
    train_text_encoder_epochs: int
    text_encoder_learning_rate: float
    train_unet: bool
    train_unet_epochs: int
    unet_learning_rate: float
    loss_function: LossFunction
    offset_noise_weight: float
    train_device: str
    temp_device: str
    train_dtype: DataType
    only_cache: bool
    resolution: int
    masked_training: bool
    unmasked_probability: float
    unmasked_weight: float
    normalize_masked_area_loss: bool
    max_noising_strength: float
    token_count: int
    initial_embedding_text: str
    lora_rank: int
    lora_alpha: float

    # sample settings
    sample_definition_file_name: str
    sample_after: float
    sample_after_unit: TimeUnit

    # backup settings
    backup_after: float
    backup_after_unit: TimeUnit
    backup_before_save: bool

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    def to_json(self):
        data = {}
        for (key, value) in vars(self).items():
            if isinstance(value, str):
                data[key] = value
            elif isinstance(value, Enum):
                data[key] = str(value)
            elif isinstance(value, bool):
                data[key] = value
            elif isinstance(value, int):
                data[key] = value
            elif isinstance(value, float):
                data[key] = value
            else:
                data[key] = value

        return data

    def from_json(self, data):
        for (key, value) in vars(self).items():
            try:
                if isinstance(value, str):
                    setattr(self, key, data[key])
                elif isinstance(value, Enum):
                    enum_type = type(getattr(self, key))
                    setattr(self, key, enum_type[data[key]])
                elif isinstance(value, bool):
                    setattr(self, key, data[key])
                elif isinstance(value, int):
                    setattr(self, key, int(data[key]))
                elif isinstance(value, float):
                    setattr(self, key, float(data[key]))
                else:
                    setattr(self, key, data[key])
            except Exception as e:
                if key in data:
                    print(f"Could not set {key} as {str(data[key])}")
                else:
                    print(f"Could not set {key}, not found.")

        print("")

    def __to_arg_name(self, var_name: str) -> str:
        return "--" + var_name.replace('_', '-')

    def __to_var_name(self, arg_name: str) -> str:
        return arg_name.lstrip('-').replace('-', '_')

    def to_args(self) -> str:
        data = []
        for (key, value) in vars(self).items():
            if isinstance(value, str):
                data.append(f"{self.__to_arg_name(key)}=\"{value}\"")
            elif isinstance(value, Enum):
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif isinstance(value, bool):
                if value:
                    data.append(self.__to_arg_name(key))
            elif isinstance(value, int):
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            elif isinstance(value, float):
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")
            else:
                data.append(f"{self.__to_arg_name(key)}=\"{str(value)}\"")

        return ' '.join(data)

    @staticmethod
    def parse_args() -> 'TrainArgs':
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        # @formatter:off

        parser.add_argument("--training-method", type=TrainingMethod, required=True, dest="training_method", help="The method of training", choices=list(TrainingMethod))
        parser.add_argument("--debug-mode", required=False, action='store_true', dest="debug_mode", help="Enable debug mode")
        parser.add_argument("--debug-dir", type=str, required=False, default="debug", dest="debug_dir", help="directory to save debug information")
        parser.add_argument("--workspace-dir", type=str, required=True, dest="workspace_dir", help="directory to use as a workspace")
        parser.add_argument("--cache-dir", type=str, required=True, dest="cache_dir", help="The directory used for caching")
        parser.add_argument("--tensorboard", required=False, action='store_true', dest="tensorboard", help="Start a tensorboard interface during training. The web server will run on port 6006")

        # model settings
        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--base-model-name", type=str, required=True, dest="base_model_name", help="The base model to start training from")
        parser.add_argument("--extra-model-name", type=str, required=False, default=None, dest="extra_model_name", help="The extra model to start training from")
        parser.add_argument("--output-dtype", type=DataType, required=True, dest="output_dtype", help="The data type to use for saving weights", choices=list(DataType))
        parser.add_argument("--output-model-format", type=ModelFormat, required=False, default=ModelFormat.CKPT, dest="output_model_format", help="The format to save the final output model", choices=list(ModelFormat))
        parser.add_argument("--output-model-destination", type=str, required=True, dest="output_model_destination", help="The destination to save the final output model")

        # data settings
        parser.add_argument("--concept-file-name", type=str, required=True, dest="concept_file_name", help="The json file containing the concept definition")
        parser.add_argument("--circular-mask-generation", required=False, action='store_true', dest="circular_mask_generation", help="Automatically generate circular masks for training")
        parser.add_argument("--random-rotate-and-crop", required=False, action='store_true', dest="random_rotate_and_crop", help="Randomly rotate and crop samples")
        parser.add_argument("--aspect-ratio-bucketing", required=False, action='store_true', dest="aspect_ratio_bucketing", help="Enable aspect ratio bucketing")
        parser.add_argument("--latent-caching", required=False, action='store_true', dest="latent_caching", help="Enable latent caching")
        parser.add_argument("--latent-caching-epochs", type=int, required=False, default=1, dest="latent_caching_epochs", help="The amount of epochs to cache, to increase sample diversity")

        # training settings
        parser.add_argument("--optimizer", type=Optimizer, required=False, default=Optimizer.ADAMW, dest="optimizer", help="The optimizer", choices=list(Optimizer))
        parser.add_argument("--learning-rate", type=float, required=False, default=3e-6, dest="learning_rate", help="The learning rate used when creating the optimizer")
        parser.add_argument("--weight-decay", type=float, required=False, default=1e-2, dest="weight_decay", help="The weight decay used when creating the optimizer")
        parser.add_argument("--loss-function", type=LossFunction, required=False, default=LossFunction.MSE, dest="loss_function", help="The loss function", choices=list(LossFunction))
        parser.add_argument("--epochs", type=int, required=True, dest="epochs", help="Number of epochs to train")
        parser.add_argument("--batch-size", type=int, required=True, dest="batch_size", help="The batch size")
        parser.add_argument("--gradient-accumulation-steps", type=int, required=False, default=1, dest="gradient_accumulation_steps", help="The amount of steps used for gradient accumulation")
        parser.add_argument("--train-text-encoder", required=False, action='store_true', dest="train_text_encoder", help="Whether the text encoder should be trained")
        parser.add_argument("--train-text-encoder-epochs", type=int, required=False, default=2 ** 30, dest="train_text_encoder_epochs", help="Number of epochs to train the text encoder for")
        parser.add_argument("--text-encoder-learning-rate", type=float, required=False, default=None, dest="text_encoder_learning_rate", help="Learning rate for the text encoder")
        parser.add_argument("--train-unet", required=False, action='store_true', dest="train_unet", help="Whether the unet should be trained")
        parser.add_argument("--train-unet-epochs", type=int, required=False, default=2 ** 30, dest="train_unet_epochs", help="Number of epochs to train the unet for")
        parser.add_argument("--unet-learning-rate", type=float, required=False, default=None, dest="unet_learning_rate", help="Learning rate for the unet")
        parser.add_argument("--offset-noise-weight", type=float, required=False, default=0.0, dest="offset_noise_weight", help="The weight for offset noise prediction")
        parser.add_argument("--train-device", type=str, required=False, default="cuda", dest="train_device", help="The device to train on")
        parser.add_argument("--temp-device", type=str, required=False, default="cpu", dest="temp_device", help="The device to use for temporary data")
        parser.add_argument("--train-dtype", type=DataType, required=False, default="float16", dest="train_dtype", help="The data type to use for training weights", choices=list(DataType))
        parser.add_argument("--only-cache", required=False, action='store_true', dest="only_cache", help="Only do the caching process without any training")
        parser.add_argument("--resolution", type=int, required=True, dest="resolution", help="Resolution to train at")
        parser.add_argument("--masked-training", required=False, action='store_true', dest="masked_training", help="Activates masked training to let the model focus on certain parts of the training sample")
        parser.add_argument("--unmasked-probability", type=float, required=False, default=0.0, dest="unmasked_probability", help="If masked training is active, defines the number of steps to train on unmasked samples")
        parser.add_argument("--unmasked-weight", type=float, required=False, default=0.0, dest="unmasked_weight", help="If masked training is active, defines the loss weight of the unmasked parts of the image")
        parser.add_argument("--normalize-masked-area-loss", required=False, action='store_true', dest="normalize_masked_area_loss", help="If masked training is active, normalizes the loss based on the masked region for each sample")
        parser.add_argument("--max-noising-strength", type=float, required=False, default=1.0, dest="max_noising_strength", help="The max noising strength for training. Useful to prevent overfitting")
        parser.add_argument("--token-count", type=int, required=False, default=1, dest="token_count", help="The number of tokens to train")
        parser.add_argument("--initial-embedding-text", type=str, required=False, default="*", dest="initial_embedding_text", help="The text to initialize new embeddings")
        parser.add_argument("--lora-rank", type=int, required=False, default=1, dest="lora_rank", help="The rank parameter used when initializing new LoRA networks")
        parser.add_argument("--lora-alpha", type=float, required=False, default=1.0, dest="lora_alpha", help="The alpha parameter used when initializing new LoRA networks")

        # sample settings
        parser.add_argument("--sample-definition-file-name", type=str, required=True, dest="sample_definition_file_name", help="The json file containing the sample definition")
        parser.add_argument("--sample-after", type=float, required=True, dest="sample_after", help="The interval to sample")
        parser.add_argument("--sample-after-unit", type=TimeUnit, required=True, dest="sample_after_unit", help="The unit applied to the sample-after option")

        # backup settings
        parser.add_argument("--backup-after", type=float, required=True, dest="backup_after", help="The interval for backups")
        parser.add_argument("--backup-after-unit", type=TimeUnit, required=True, dest="backup_after_unit", help="The unit applied to the backup-after option")
        parser.add_argument("--backup-before-save", required=False, action='store_true', dest="backup_before_save", help="Create a backup before saving the final model")

        # @formatter:on

        return TrainArgs(vars(parser.parse_args()))

    @staticmethod
    def default_values():
        args = {}

        args["training_method"] = TrainingMethod.FINE_TUNE
        args["debug_mode"] = False
        args["debug_dir"] = "debug"
        args["workspace_dir"] = "workspace/run"
        args["cache_dir"] = "workspace-cache/run"
        args["tensorboard"] = True

        # model settings
        args["model_type"] = ModelType.STABLE_DIFFUSION_15
        args["base_model_name"] = ""
        args["extra_model_name"] = ""
        args["output_dtype"] = DataType.FLOAT_32
        args["output_model_format"] = ModelFormat.SAFETENSORS
        args["output_model_destination"] = "models/model.safetensors"

        # data settings
        args["concept_file_name"] = "training_concepts/concepts.json"
        args["circular_mask_generation"] = False
        args["random_rotate_and_crop"] = False
        args["aspect_ratio_bucketing"] = True
        args["latent_caching"] = True
        args["latent_caching_epochs"] = 1

        # training settings
        args["optimizer"] = Optimizer.ADAMW
        args["learning_rate"] = 3e-6
        args["weight_decay"] = 1e-2
        args["loss_function"] = LossFunction.MSE
        args["epochs"] = 100
        args["batch_size"] = 1
        args["gradient_accumulation_steps"] = 1
        args["train_text_encoder"] = True
        args["train_text_encoder_epochs"] = 30
        args["text_encoder_learning_rate"] = 3e-6
        args["train_unet"] = True
        args["train_unet_epochs"] = 100
        args["unet_learning_rate"] = 3e-6
        args["offset_noise_weight"] = 0.0
        args["train_device"] = "cuda"
        args["temp_device"] = "cpu"
        args["train_dtype"] = DataType.FLOAT_16
        args["only_cache"] = False
        args["resolution"] = 512
        args["masked_training"] = False
        args["unmasked_probability"] = 0.1
        args["unmasked_weight"] = 0.1
        args["normalize_masked_area_loss"] = True
        args["max_noising_strength"] = 1.0
        args["token_count"] = 1
        args["initial_embedding_text"] = "*"
        args["lora_rank"] = 16
        args["lora_alpha"] = 1.0

        # sample settings
        args["sample_definition_file_name"] = "training_samples/samples.json"
        args["sample_after"] = 10
        args["sample_after_unit"] = TimeUnit.MINUTE

        # backup settings
        args["backup_after"] = 30
        args["backup_after_unit"] = TimeUnit.MINUTE
        args["backup_before_save"] = True

        return TrainArgs(args)
