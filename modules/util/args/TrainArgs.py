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
    circular_mask_generation: bool
    random_rotate_and_crop: bool
    aspect_ratio_bucketing: bool
    latent_caching: bool

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
    resolution: int
    masked_training: bool
    unmasked_probability: float
    unmasked_weight: float
    normalize_masked_area_loss: bool
    max_noising_strength: float

    # sample settings
    sample_prompt: str
    sample_after: float
    sample_after_unit: TimeUnit
    sample_dir: str
    sample_resolution: int

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
        parser.add_argument("--circular-mask-generation", required=False, action='store_true', dest="circular_mask_generation", help="Automatically generate circular masks for training")
        parser.add_argument("--random-rotate-and-crop", required=False, action='store_true', dest="random_rotate_and_crop", help="Randomly rotate and crop samples")
        parser.add_argument("--aspect-ratio-bucketing", required=False, action='store_true', dest="aspect_ratio_bucketing", help="Enable aspect ratio bucketing")
        parser.add_argument("--latent-caching", required=False, action='store_true', dest="latent_caching", help="Enable latent caching")

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
        parser.add_argument("--resolution", type=int, required=True, dest="resolution", help="Resolution to train at")
        parser.add_argument("--masked-training", required=False, action='store_true', dest="masked_training", help="Aktivates masked training to let the model focus on certain parts of the training sample")
        parser.add_argument("--unmasked-probability", type=float, required=False, default=0, dest="unmasked_probability", help="If masked training is active, defines the number of steps to train on unmasked samples")
        parser.add_argument("--unmasked-weight", type=float, required=False, default=0, dest="unmasked_weight", help="If masked training is active, defines the loss weight of the unmasked parts of the image")
        parser.add_argument("--normalize-masked-area-loss", required=False, action='store_true', dest="normalize_masked_area_loss", help="If masked training is active, normalizes the loss based on the masked region for each sample")
        parser.add_argument("--max-noising-strength", type=float, required=False, default=1, dest="max_noising_strength", help="The max noising strength for training. Useful to prevent overfitting")

        # sample settings
        parser.add_argument("--sample-prompt", type=str, required=True, dest="sample_prompt", help="The prompt used for sampling")
        parser.add_argument("--sample-after", type=float, required=True, dest="sample_after", help="The interval to sample")
        parser.add_argument("--sample-after-unit", type=TimeUnit, required=True, dest="sample_after_unit", help="The unit applied to the sample-after option")
        parser.add_argument("--sample-dir", type=str, required=True, dest="sample_dir", help="Directory to save samples")
        parser.add_argument("--sample-resolution", type=int, required=False, default=512, dest="sample_resolution", help="The resolution of samples")

        return TrainArgs(vars(parser.parse_args()))
