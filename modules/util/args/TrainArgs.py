import argparse

from modules.util.args.arg_type_util import *
from modules.util.enum.LossFunction import LossFunction
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod


class TrainArgs:
    training_method: TrainingMethod
    debug_mode: bool
    debug_dir: str

    # model settings
    model_type: ModelType
    base_model_name: str
    output_dtype: torch.dtype

    # data settings
    concept_file_name: str
    output_model_format: ModelFormat
    output_model_destination: str

    # training settings
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    train_text_encoder: bool
    train_text_encoder_epochs: int
    loss_function: LossFunction
    offset_noise_weight: float
    train_device: torch.device
    temp_device: torch.device
    train_dtype: torch.dtype
    cache_dir: str
    normalize_masked_area_loss: bool
    resolution: int

    # sample settings
    sample_prompt: str
    sample_after: float
    sample_after_unit: TimeUnit
    sample_dir: str

    def __init__(self, args: dict):
        for (key, value) in args.items():
            setattr(self, key, value)

    @staticmethod
    def parse_args() -> 'TrainArgs':
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        parser.add_argument("--training-method", type=TrainingMethod, required=True, dest="training_method", help="The method of training", choices=list(TrainingMethod))
        parser.add_argument("--debug-mode", required=False, action='store_true', dest="debug_mode", help="Enable debug mode")
        parser.add_argument("--debug-dir", type=str, required=False, default="debug", dest="debug_dir", help="directory to save debug information")

        # model settings
        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--base-model-name", type=str, required=True, dest="base_model_name", help="The base model to start training from")
        parser.add_argument("--output-dtype", type=torch_dtype, required=True, dest="output_dtype", help="The data type to use for saving weights")

        # data settings
        parser.add_argument("--concept-file-name", type=str, required=True, dest="concept_file_name", help="The json file containing the concept definition")
        parser.add_argument("--output-model-format", type=ModelFormat, required=False, default=ModelFormat.CKPT, dest="output_model_format", help="The format to save the final output model", choices=list(ModelFormat))
        parser.add_argument("--output-model-destination", type=str, required=True, dest="output_model_destination", help="The destination to save the final output model")

        # training settings
        parser.add_argument("--epochs", type=int, required=True, dest="epochs", help="Number of epochs to train")
        parser.add_argument("--batch-size", type=int, required=True, dest="batch_size", help="The batch size")
        parser.add_argument("--gradient-accumulation-steps", type=int, required=False, default=1, dest="gradient_accumulation_steps", help="The amount of steps used for gradient accumulation")
        parser.add_argument("--train-text-encoder", required=False, action='store_true', dest="train_text_encoder", help="Whether the text encoder should be trained")
        parser.add_argument("--train-text-encoder-epochs", type=int, required=True, dest="train_text_encoder_epochs", help="Number of epochs to train the text encoder for")
        parser.add_argument("--loss-function", type=LossFunction, required=False, default=LossFunction.MSE, dest="loss_function", help="The loss function", choices=list(LossFunction))
        parser.add_argument("--offset_noise_weight", type=float, required=False, default=0, dest="offset_noise_weight", help="The weight for offset noise prediction")
        parser.add_argument("--train-device", type=torch_device, required=False, default="cuda", dest="train_device", help="The device to train on")
        parser.add_argument("--temp-device", type=torch_device, required=False, default="cpu", dest="temp_device", help="The device to use for temporary data")
        parser.add_argument("--train-dtype", type=torch_dtype, required=False, default="float16", dest="train_dtype", help="The data type to use for training weights")
        parser.add_argument("--cache-dir", type=str, required=True, dest="cache_dir", help="The directory used for caching")
        parser.add_argument("--learning-rate", type=float, required=True, dest="learning_rate", help="The learning rate")
        parser.add_argument("--normalize-masked-area-loss", required=False, action='store_true', dest="normalize_masked_area_loss", help="Normalizes the loss based on the masked region")
        parser.add_argument("--resolution", type=int, required=True, dest="resolution", help="Resolution to train at")

        # sample settings
        parser.add_argument("--sample-prompt", type=str, required=True, dest="sample_prompt", help="The prompt used for sampling")
        parser.add_argument("--sample-after", type=float, required=True, dest="sample_after", help="The interval to sample")
        parser.add_argument("--sample-after-unit", type=TimeUnit, required=True, dest="sample_after_unit", help="The unit applied to the sample-after option")
        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory to save samples")

        return TrainArgs(vars(parser.parse_args()))
