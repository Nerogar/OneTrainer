import argparse

from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod


class TrainArgs(BaseArgs):
    training_method: TrainingMethod
    model_type: ModelType
    debug_mode: bool
    debug_dir: str
    workspace_dir: str
    cache_dir: str
    tensorboard: bool
    continue_last_backup: bool

    # model settings
    base_model_name: str
    extra_model_name: str
    weight_dtype: DataType
    text_encoder_weight_dtype: DataType
    unet_weight_dtype: DataType
    vae_weight_dtype: DataType
    output_dtype: DataType
    output_model_format: ModelFormat
    output_model_destination: str
    gradient_checkpointing: bool

    # data settings
    concept_file_name: str
    circular_mask_generation: bool
    random_rotate_and_crop: bool
    aspect_ratio_bucketing: bool
    latent_caching: bool
    latent_caching_epochs: int
    clear_cache_before_training: bool

    # training settings
    optimizer: Optimizer
    learning_rate_scheduler: LearningRateScheduler
    learning_rate: float
    learning_rate_warmup_steps: int
    learning_rate_cycles: float
    weight_decay: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    ema: EMAMode
    ema_decay: float
    ema_update_step_interval: int
    train_text_encoder: bool
    train_text_encoder_epochs: int
    text_encoder_learning_rate: float
    text_encoder_layer_skip: int
    train_unet: bool
    train_unet_epochs: int
    unet_learning_rate: float
    offset_noise_weight: float
    perturbation_noise_weight: float
    rescale_noise_scheduler_to_zero_terminal_snr: bool
    force_v_prediction: bool
    force_epsilon_prediction: bool
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
    embedding_weight_dtype: DataType
    lora_rank: int
    lora_alpha: float
    lora_weight_dtype: DataType
    attention_mechanism: AttentionMechanism

    # sample settings
    sample_definition_file_name: str
    sample_after: float
    sample_after_unit: TimeUnit
    sample_image_format: ImageFormat

    # backup settings
    backup_after: float
    backup_after_unit: TimeUnit
    rolling_backup: bool
    rolling_backup_count: int
    backup_before_save: bool
    save_after: float
    save_after_unit: TimeUnit

    def __init__(self, args: dict):
        super(TrainArgs, self).__init__(args)

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes(
            self.weight_dtype if self.text_encoder_weight_dtype == DataType.NONE else self.text_encoder_weight_dtype,
            self.weight_dtype if self.unet_weight_dtype == DataType.NONE else self.unet_weight_dtype,
            self.weight_dtype if self.vae_weight_dtype == DataType.NONE else self.vae_weight_dtype,
            self.weight_dtype if self.lora_weight_dtype == DataType.NONE else self.lora_weight_dtype,
            self.weight_dtype if self.embedding_weight_dtype == DataType.NONE else self.embedding_weight_dtype,
        )

    def trainable_weight_dtypes(self) -> list[DataType]:
        weight_dtypes = self.weight_dtypes()

        if self.training_method == TrainingMethod.LORA:
            return [weight_dtypes.lora]
        elif self.training_method == TrainingMethod.EMBEDDING:
            return [weight_dtypes.embedding]
        elif self.training_method == TrainingMethod.FINE_TUNE_VAE:
            return [weight_dtypes.vae]
        elif self.training_method == TrainingMethod.FINE_TUNE:
            dtypes = []
            if self.train_text_encoder:
                dtypes.append(weight_dtypes.text_encoder)
            elif self.unet_weight_dtype:
                dtypes.append(weight_dtypes.unet)
            return dtypes

    @staticmethod
    def parse_args() -> 'TrainArgs':
        parser = argparse.ArgumentParser(description="One Trainer Training Script.")

        # @formatter:off

        parser.add_argument("--training-method", type=TrainingMethod, required=True, dest="training_method", help="The method of training", choices=list(TrainingMethod))
        parser.add_argument("--model-type", type=ModelType, required=True, dest="model_type", help="Type of the base model", choices=list(ModelType))
        parser.add_argument("--debug-mode", required=False, action='store_true', dest="debug_mode", help="Enable debug mode")
        parser.add_argument("--debug-dir", type=str, required=False, default="debug", dest="debug_dir", help="directory to save debug information")
        parser.add_argument("--workspace-dir", type=str, required=True, dest="workspace_dir", help="directory to use as a workspace")
        parser.add_argument("--cache-dir", type=str, required=True, dest="cache_dir", help="The directory used for caching")
        parser.add_argument("--tensorboard", required=False, action='store_true', dest="tensorboard", help="Start a tensorboard interface during training. The web server will run on port 6006")
        parser.add_argument("--continue-last-backup", required=False, action='store_true', dest="continue_last_backup", help="Continues training from the last backup in <workspace>/run/backup")

        # model settings
        parser.add_argument("--base-model-name", type=str, required=True, dest="base_model_name", help="The base model to start training from")
        parser.add_argument("--extra-model-name", type=str, required=False, default=None, dest="extra_model_name", help="The extra model to start training from")
        parser.add_argument("--weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="weight_dtype", help="The data type to use for weights during training", choices=list(DataType))
        parser.add_argument("--text-encoder-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="text_encoder_weight_dtype", help="The data type to use for text encoder weights during training", choices=list(DataType))
        parser.add_argument("--unet-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="unet_weight_dtype", help="The data type to use for unet weights during training", choices=list(DataType))
        parser.add_argument("--vae-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="vae_weight_dtype", help="The data type to use for vae weights during training", choices=list(DataType))
        parser.add_argument("--output-dtype", type=DataType, required=True, dest="output_dtype", help="The data type to use for saving weights", choices=list(DataType))
        parser.add_argument("--output-model-format", type=ModelFormat, required=False, default=ModelFormat.SAFETENSORS, dest="output_model_format", help="The format to save the final output model", choices=list(ModelFormat))
        parser.add_argument("--output-model-destination", type=str, required=True, dest="output_model_destination", help="The destination to save the final output model")
        parser.add_argument("--gradient-checkpointing", required=False, action='store_true', dest="gradient_checkpointing", help="Enable gradient checkpointing to reduce memory usage")

        # data settings
        parser.add_argument("--concept-file-name", type=str, required=True, dest="concept_file_name", help="The json file containing the concept definition")
        parser.add_argument("--circular-mask-generation", required=False, action='store_true', dest="circular_mask_generation", help="Automatically generate circular masks for training")
        parser.add_argument("--random-rotate-and-crop", required=False, action='store_true', dest="random_rotate_and_crop", help="Randomly rotate and crop samples")
        parser.add_argument("--aspect-ratio-bucketing", required=False, action='store_true', dest="aspect_ratio_bucketing", help="Enable aspect ratio bucketing")
        parser.add_argument("--latent-caching", required=False, action='store_true', dest="latent_caching", help="Enable latent caching")
        parser.add_argument("--latent-caching-epochs", type=int, required=False, default=1, dest="latent_caching_epochs", help="The amount of epochs to cache, to increase sample diversity")
        parser.add_argument("--clear-cache-before-training", required=False, action='store_true', dest="clear_cache_before_training", help="Clears the latent cache before starting to train")

        # training settings
        parser.add_argument("--optimizer", type=Optimizer, required=False, default=Optimizer.ADAMW, dest="optimizer", help="The optimizer", choices=list(Optimizer))
        parser.add_argument("--learning-rate-scheduler", type=LearningRateScheduler, required=False, default=LearningRateScheduler.CONSTANT, dest="learning_rate_scheduler", help="The learning rate scheduler")
        parser.add_argument("--learning-rate", type=float, required=False, default=3e-6, dest="learning_rate", help="The learning rate used when creating the optimizer")
        parser.add_argument("--learning-rate-warmup-steps", type=int, required=False, default=0, dest="learning_rate_warmup_steps", help="The number of warmup steps when creating the learning rate scheduler")
        parser.add_argument("--learning-rate-cycles", type=float, required=False, default=1, dest="learning_rate_cycles", help="The number of cycles of the learning rate scheduler")
        parser.add_argument("--weight-decay", type=float, required=False, default=1e-2, dest="weight_decay", help="The weight decay used when creating the optimizer")
        parser.add_argument("--epochs", type=int, required=True, dest="epochs", help="Number of epochs to train")
        parser.add_argument("--batch-size", type=int, required=True, dest="batch_size", help="The batch size")
        parser.add_argument("--gradient-accumulation-steps", type=int, required=False, default=1, dest="gradient_accumulation_steps", help="The amount of steps used for gradient accumulation")
        parser.add_argument("--ema", type=EMAMode, required=False, default=EMAMode.OFF, dest="ema", help="Activate EMA during training", choices=list(EMAMode))
        parser.add_argument("--ema-decay", type=float, required=False, default=0.999, dest="ema_decay", help="Decay parameter of the EMA model")
        parser.add_argument("--ema-update-step-interval", type=int, required=False, default=5, dest="ema_update_step_interval", help="")
        parser.add_argument("--train-text-encoder", required=False, action='store_true', dest="train_text_encoder", help="Whether the text encoder should be trained")
        parser.add_argument("--train-text-encoder-epochs", type=int, required=False, default=2 ** 30, dest="train_text_encoder_epochs", help="Number of epochs to train the text encoder for")
        parser.add_argument("--text-encoder-learning-rate", type=float, required=False, default=None, dest="text_encoder_learning_rate", help="Learning rate for the text encoder")
        parser.add_argument("--text-encoder-layer-skip", type=int, required=False, default=0, dest="text_encoder_layer_skip", help="Skip last layers of the text encoder")
        parser.add_argument("--train-unet", required=False, action='store_true', dest="train_unet", help="Whether the unet should be trained")
        parser.add_argument("--train-unet-epochs", type=int, required=False, default=2 ** 30, dest="train_unet_epochs", help="Number of epochs to train the unet for")
        parser.add_argument("--unet-learning-rate", type=float, required=False, default=None, dest="unet_learning_rate", help="Learning rate for the unet")
        parser.add_argument("--offset-noise-weight", type=float, required=False, default=0.0, dest="offset_noise_weight", help="The weight for offset noise prediction")
        parser.add_argument("--perturbation-noise-weight", type=float, required=False, default=0.0, dest="perturbation_noise_weight", help="The weight for perturbation noise")
        parser.add_argument("--rescale-noise-scheduler-to-zero-terminal-snr", required=False, action='store_true', dest="rescale_noise_scheduler_to_zero_terminal_snr", help="Rescales the noise sceduler to have a zero terminal signal to noise ratio, this also sets the model to v-prediction mode")
        parser.add_argument("--force-v-prediction", required=False, action='store_true', dest="force_v_prediction", help="Forces the training to use v-prediction")
        parser.add_argument("--force-epsilon-prediction", required=False, action='store_true', dest="force_epsilon_prediction", help="Forces the training to use epsilon-prediction")
        parser.add_argument("--train-device", type=str, required=False, default="cuda", dest="train_device", help="The device to train on")
        parser.add_argument("--temp-device", type=str, required=False, default="cpu", dest="temp_device", help="The device to use for temporary data")
        parser.add_argument("--train-dtype", type=DataType, required=False, default=DataType.FLOAT_16, dest="train_dtype", help="The data type to use for training weights", choices=list(DataType))
        parser.add_argument("--only-cache", required=False, action='store_true', dest="only_cache", help="Only do the caching process without any training")
        parser.add_argument("--resolution", type=int, required=True, dest="resolution", help="Resolution to train at")
        parser.add_argument("--masked-training", required=False, action='store_true', dest="masked_training", help="Activates masked training to let the model focus on certain parts of the training sample")
        parser.add_argument("--unmasked-probability", type=float, required=False, default=0.0, dest="unmasked_probability", help="If masked training is active, defines the number of steps to train on unmasked samples")
        parser.add_argument("--unmasked-weight", type=float, required=False, default=0.0, dest="unmasked_weight", help="If masked training is active, defines the loss weight of the unmasked parts of the image")
        parser.add_argument("--normalize-masked-area-loss", required=False, action='store_true', dest="normalize_masked_area_loss", help="If masked training is active, normalizes the loss based on the masked region for each sample")
        parser.add_argument("--max-noising-strength", type=float, required=False, default=1.0, dest="max_noising_strength", help="The max noising strength for training. Useful to prevent overfitting")
        parser.add_argument("--token-count", type=int, required=False, default=1, dest="token_count", help="The number of tokens to train")
        parser.add_argument("--initial-embedding-text", type=str, required=False, default="*", dest="initial_embedding_text", help="The text to initialize new embeddings")
        parser.add_argument("--embedding-weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="embedding_weight_dtype", help="The data type to use for training the Embedding", choices=list(DataType))
        parser.add_argument("--lora-rank", type=int, required=False, default=1, dest="lora_rank", help="The rank parameter used when initializing new LoRA networks")
        parser.add_argument("--lora-alpha", type=float, required=False, default=1.0, dest="lora_alpha", help="The alpha parameter used when initializing new LoRA networks")
        parser.add_argument("--lora-weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="lora_weight_dtype", help="The data type to use for training the LoRA", choices=list(DataType))
        parser.add_argument("--attention-mechanism", type=AttentionMechanism, required=False, default=AttentionMechanism.XFORMERS, dest="attention_mechanism", help="The Attention mechanism to use", choices=list(AttentionMechanism))

        # sample settings
        parser.add_argument("--sample-definition-file-name", type=str, required=True, dest="sample_definition_file_name", help="The json file containing the sample definition")
        parser.add_argument("--sample-after", type=float, required=True, dest="sample_after", help="The interval to sample")
        parser.add_argument("--sample-after-unit", type=TimeUnit, required=True, dest="sample_after_unit", help="The unit applied to the sample-after option")
        parser.add_argument("--sample-image-format", type=ImageFormat, required=False, default=ImageFormat.JPG, dest="sample_image_format", help="The file format used when saving samples", choices=list(ImageFormat))

        # backup settings
        parser.add_argument("--backup-after", type=float, required=True, dest="backup_after", help="The interval for backups")
        parser.add_argument("--backup-after-unit", type=TimeUnit, required=True, dest="backup_after_unit", help="The unit applied to the backup-after option")
        parser.add_argument("--rolling-backup", required=False, action='store_true', dest="rolling_backup", help="Enable rolling backups")
        parser.add_argument("--rolling-backup-count", type=int, required=False, default=3, dest="rolling_backup_count", help="The number of backups to keep if rolling backups are enabled")
        parser.add_argument("--backup-before-save", required=False, action='store_true', dest="backup_before_save", help="Create a backup before saving the final model")
        parser.add_argument("--save-after", type=float, required=False, default=0, dest="save_after", help="The interval for backups")
        parser.add_argument("--save-after-unit", type=TimeUnit, required=False, default=TimeUnit.NEVER, dest="save_after_unit", help="The unit applied to the backup-after option")

        # @formatter:on

        return TrainArgs(vars(parser.parse_args()))

    @staticmethod
    def default_values():
        args = {}

        args["training_method"] = TrainingMethod.FINE_TUNE
        args["model_type"] = ModelType.STABLE_DIFFUSION_15
        args["debug_mode"] = False
        args["debug_dir"] = "debug"
        args["workspace_dir"] = "workspace/run"
        args["cache_dir"] = "workspace-cache/run"
        args["tensorboard"] = True
        args["continue_last_backup"] = False

        # model settings
        args["base_model_name"] = "runwayml/stable-diffusion-v1-5"
        args["extra_model_name"] = ""
        args["weight_dtype"] = DataType.FLOAT_32
        args["text_encoder_weight_dtype"] = DataType.NONE
        args["unet_weight_dtype"] = DataType.NONE
        args["vae_weight_dtype"] = DataType.FLOAT_32
        args["output_dtype"] = DataType.FLOAT_32
        args["output_model_format"] = ModelFormat.SAFETENSORS
        args["output_model_destination"] = "models/model.safetensors"
        args["gradient_checkpointing"] = True

        # data settings
        args["concept_file_name"] = "training_concepts/concepts.json"
        args["circular_mask_generation"] = False
        args["random_rotate_and_crop"] = False
        args["aspect_ratio_bucketing"] = True
        args["latent_caching"] = True
        args["latent_caching_epochs"] = 1
        args["clear_cache_before_training"] = True

        # training settings
        args["epochs"] = 100
        args["batch_size"] = 1
        args["gradient_accumulation_steps"] = 1
        args["ema"] = EMAMode.OFF
        args["ema_decay"] = 0.999
        args["ema_update_step_interval"] = 5
        args["text_encoder_layer_skip"] = 0
        args["offset_noise_weight"] = 0.0
        args["perturbation_noise_weight"] = 0.0
        args["rescale_noise_scheduler_to_zero_terminal_snr"] = False
        args["force_v_prediction"] = False
        args["force_epsilon_prediction"] = False
        args["train_device"] = "cuda"
        args["temp_device"] = "cpu"
        args["train_dtype"] = DataType.FLOAT_16
        args["only_cache"] = False
        args["resolution"] = 512
        args["masked_training"] = False
        args["unmasked_probability"] = 0.1
        args["unmasked_weight"] = 0.1
        args["normalize_masked_area_loss"] = False
        args["max_noising_strength"] = 1.0
        args["token_count"] = 1
        args["initial_embedding_text"] = "*"
        args["embedding_weight_dtype"] = DataType.FLOAT_32
        args["lora_rank"] = 16
        args["lora_alpha"] = 1.0
        args["lora_weight_dtype"] = DataType.FLOAT_32
        args["attention_mechanism"] = AttentionMechanism.XFORMERS
        
        # optimizer settings
        args["optimizer"] = Optimizer.ADAMW
        args["weight_decay"] = 1e-2
        args["momentum"] = 0.99
        args["dampening"] = 0
        args["nesterov"] = False
        args["eps"] = 1e-8
        args["foreach"] = False  # Disabled, because it uses too much VRAM
        args["fused"] = True
        args["min_8bit_size"] = 4096
        args["percentile_clipping"] = 100
        args["block_wise"] = True
        args["is_paged"] = False
        args["lr_decay"] = 0
        args["initial_accumulator_value"] = 0
        args["alpha"] = 0.99
        args["centered"] = False
        args["max_unorm"] = 0.02
        args["betas"] = (0.999, 0.999)
        args["bias_correction"] = True
        args["amsgrad"] = False
        args["adam_w_mode"] = True
        args["use_bias_correction"] = True
        args["safeguard_warmup"] = True
        args["beta3"] = None
        args["decouple"] = False
        args["d0"] = 1e-6
        args["d_coef"] = 1.0
        args["growth_rate"] = float('inf')
        args["fsdp_in_use"] = False
        args["clip_threshold"] = 1.0
        args["decay_rate"] = -0.8
        args["beta1"] = None
        args["weight_decay"] = 0.0
        args["scale_parameter"] = True
        args["relative_step"] = True
        args["warmup_init"] = False
        args["eps_tuple"] = (1e-30, 1e-3)
        # sample settings
        args["sample_definition_file_name"] = "training_samples/samples.json"
        args["sample_after"] = 10
        args["sample_after_unit"] = TimeUnit.MINUTE
        args["sample_image_format"] = ImageFormat.JPG

        # backup settings
        args["backup_after"] = 30
        args["backup_after_unit"] = TimeUnit.MINUTE
        args["rolling_backup"] = False
        args["rolling_backup_count"] = 3
        args["backup_before_save"] = True
        args["save_after"] = 0
        args["save_after_unit"] = TimeUnit.NEVER

        return TrainArgs(args)
