import argparse
from typing import Any

from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.args.BaseArgs import BaseArgs
from modules.util.args.arg_type_util import nullable_bool
from modules.util.enum.AlignPropLoss import AlignPropLoss
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
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LearningRateScaler import LearningRateScaler


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
    weight_dtype: DataType
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
    learning_rate_scheduler: LearningRateScheduler
    learning_rate: float
    learning_rate_warmup_steps: int
    learning_rate_cycles: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    ema: EMAMode
    ema_decay: float
    ema_update_step_interval: int
    train_device: str
    temp_device: str
    train_dtype: DataType
    only_cache: bool
    resolution: str
    attention_mechanism: AttentionMechanism
    align_prop: bool
    align_prop_probability: float
    align_prop_loss: AlignPropLoss
    align_prop_weight: float
    align_prop_steps: int
    align_prop_truncate_steps: float
    align_prop_cfg_scale: float
    mse_strength: float
    mae_strength: float
    loss_scaler: LossScaler
    learning_rate_scaler: LearningRateScaler

    # unet
    train_unet: bool
    train_unet_epochs: int
    unet_learning_rate: float
    offset_noise_weight: float
    perturbation_noise_weight: float
    rescale_noise_scheduler_to_zero_terminal_snr: bool
    force_v_prediction: bool
    force_epsilon_prediction: bool
    max_noising_strength: float
    unet_weight_dtype: DataType

    # prior
    train_prior: bool
    train_prior_epochs: int
    prior_learning_rate: float
    prior_weight_dtype: DataType

    # text encoder
    train_text_encoder: bool
    train_text_encoder_epochs: int
    text_encoder_learning_rate: float
    text_encoder_layer_skip: int
    text_encoder_weight_dtype: DataType

    # text encoder 2
    train_text_encoder_2: bool
    train_text_encoder_2_epochs: int
    text_encoder_2_learning_rate: float
    text_encoder_2_layer_skip: int
    text_encoder_2_weight_dtype: DataType

    # vae
    vae_weight_dtype: DataType

    # effnet encoder
    effnet_encoder_model_name: str
    effnet_encoder_weight_dtype: DataType

    # decoder
    decoder_model_name: str
    decoder_weight_dtype: DataType

    # decoder text encoder
    decoder_text_encoder_weight_dtype: DataType

    # decoder vqgan
    decoder_vqgan_weight_dtype: DataType

    # masked training
    masked_training: bool
    unmasked_probability: float
    unmasked_weight: float
    normalize_masked_area_loss: bool

    # embedding
    embedding_model_names: list[str]
    token_count: int
    initial_embedding_text: str
    embedding_weight_dtype: DataType

    # lora
    lora_model_name: str
    lora_rank: int
    lora_alpha: float
    lora_weight_dtype: DataType

    # optimizer settings
    optimizer: Optimizer
    optimizer_weight_decay: float
    optimizer_momentum: float
    optimizer_dampening: float
    optimizer_nesterov: bool
    optimizer_eps: float
    optimizer_foreach: bool
    optimizer_fused: bool
    optimizer_min_8bit_size: int
    optimizer_percentile_clipping: int
    optimizer_block_wise: bool
    optimizer_is_paged: bool
    optimizer_lr_decay: int
    optimizer_initial_accumulator_value: int
    optimizer_alpha: float
    optimizer_centered: bool
    optimizer_max_unorm: float
    optimizer_beta2: float
    optimizer_bias_correction: bool
    optimizer_amsgrad: bool
    optimizer_adam_w_mode: bool
    optimizer_use_bias_correction: bool
    optimizer_safeguard_warmup: bool
    optimizer_beta3: float
    optimizer_decouple: bool
    optimizer_d0: float
    optimizer_d_coef: float
    optimizer_growth_rate: float
    optimizer_fsdp_in_use: bool
    optimizer_clip_threshold: float
    optimizer_decay_rate: float
    optimizer_beta1: float
    optimizer_scale_parameter: bool
    optimizer_relative_step: bool
    optimizer_warmup_init: bool
    optimizer_eps2: float
    optimizer_optim_bits: int
    optimizer_log_every: int
    optimizer_no_prox: bool
    optimizer_maximize: bool
    optimizer_capturable: bool
    optimizer_differentiable: bool
    optimizer_use_triton: bool

    # sample settings
    sample_definition_file_name: str
    sample_after: float
    sample_after_unit: TimeUnit
    sample_image_format: ImageFormat
    samples_to_tensorboard: bool
    non_ema_sampling: bool

    # backup settings
    backup_after: float
    backup_after_unit: TimeUnit
    rolling_backup: bool
    rolling_backup_count: int
    backup_before_save: bool
    save_after: float
    save_after_unit: TimeUnit

    def __init__(self, data: list[(str, Any, type, bool)]):
        super(TrainArgs, self).__init__(data)

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes(
            self.weight_dtype if self.unet_weight_dtype == DataType.NONE else self.unet_weight_dtype,
            self.weight_dtype if self.prior_weight_dtype == DataType.NONE else self.prior_weight_dtype,
            self.weight_dtype if self.text_encoder_weight_dtype == DataType.NONE else self.text_encoder_weight_dtype,
            self.weight_dtype if self.text_encoder_2_weight_dtype == DataType.NONE else self.text_encoder_2_weight_dtype,
            self.weight_dtype if self.vae_weight_dtype == DataType.NONE else self.vae_weight_dtype,
            self.weight_dtype if self.effnet_encoder_weight_dtype == DataType.NONE else self.effnet_encoder_weight_dtype,
            self.weight_dtype if self.decoder_weight_dtype == DataType.NONE else self.decoder_weight_dtype,
            self.weight_dtype if self.decoder_text_encoder_weight_dtype == DataType.NONE else self.decoder_text_encoder_weight_dtype,
            self.weight_dtype if self.decoder_vqgan_weight_dtype == DataType.NONE else self.decoder_vqgan_weight_dtype,
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

    def model_names(self) -> ModelNames:
        return ModelNames(
            base_model=self.base_model_name,
            effnet_encoder_model=self.effnet_encoder_model_name,
            decoder_model=self.decoder_model_name,
            lora=self.lora_model_name,
            embedding=self.embedding_model_names,
        )

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
        parser.add_argument("--weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="weight_dtype", help="The data type to use for weights during training", choices=list(DataType))
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
        parser.add_argument("--epochs", type=int, required=True, dest="epochs", help="Number of epochs to train")
        parser.add_argument("--batch-size", type=int, required=True, dest="batch_size", help="The batch size")
        parser.add_argument("--gradient-accumulation-steps", type=int, required=False, default=1, dest="gradient_accumulation_steps", help="The amount of steps used for gradient accumulation")
        parser.add_argument("--ema", type=EMAMode, required=False, default=EMAMode.OFF, dest="ema", help="Activate EMA during training", choices=list(EMAMode))
        parser.add_argument("--ema-decay", type=float, required=False, default=0.999, dest="ema_decay", help="Decay parameter of the EMA model")
        parser.add_argument("--ema-update-step-interval", type=int, required=False, default=5, dest="ema_update_step_interval", help="")
        parser.add_argument("--train-device", type=str, required=False, default="cuda", dest="train_device", help="The device to train on")
        parser.add_argument("--temp-device", type=str, required=False, default="cpu", dest="temp_device", help="The device to use for temporary data")
        parser.add_argument("--train-dtype", type=DataType, required=False, default=DataType.FLOAT_16, dest="train_dtype", help="The data type to use for training weights", choices=list(DataType))
        parser.add_argument("--only-cache", required=False, action='store_true', dest="only_cache", help="Only do the caching process without any training")
        parser.add_argument("--resolution", type=str, required=True, dest="resolution", help="Resolution to train at")
        parser.add_argument("--attention-mechanism", type=AttentionMechanism, required=False, default=AttentionMechanism.XFORMERS, dest="attention_mechanism", help="The Attention mechanism to use", choices=list(AttentionMechanism))
        parser.add_argument("--align-prop", required=False, action='store_true', dest="align_prop", help="Enable AlignProp loss calculations")
        parser.add_argument("--align-prop-probability", type=float, required=False, default=0.1, dest="align_prop_probability", help="If AlignProp is active, defines the number of steps that use the AlignProp loss")
        parser.add_argument("--align-prop-loss", type=AlignPropLoss, required=False, default=AlignPropLoss.AESTHETIC, dest="align_prop_loss", help="The AlignProp loss function", choices=list(AlignPropLoss))
        parser.add_argument("--align-prop-weight", type=float, required=False, default=0.01, dest="align_prop_weight", help="A weight multiplier for the AlignProp loss")
        parser.add_argument("--align-prop-steps", type=int, required=False, default=20, dest="align_prop_steps", help="Number of inference steps for each AlignProp step")
        parser.add_argument("--align-prop-truncate-steps", type=float, required=False, default=0.5, dest="align_prop_truncate_steps", help="Fraction of steps to randomly truncate when using AlignProp")
        parser.add_argument("--align-prop-cfg-scale", type=float, required=False, default=7.0, dest="align_prop_cfg_scale", help="CFG Scale for inference steps of AlignProp calculations")
        parser.add_argument("--mse-strength", type=float, required=False, default=1.0, dest="mse_strength", help="Mean squared Error strength for custom loss settings")
        parser.add_argument("--mae-strength", type=float, required=False, default=0.0, dest="mae_strength", help="Mean Absolute Error strength for custom loss settings")
        parser.add_argument("--loss-scaler", type=LossScaler, required=False, default=LossScaler.NONE, dest="loss_scaler", help="Type of Loss Scaler", choices=list(LossScaler))
        parser.add_argument("--learning-rate-scaler", type=LearningRateScaler, required=False, default=LearningRateScaler.NONE, dest="learning_rate_scaler", help="Type of Learning Rate Scaler", choices=list(LearningRateScaler))

        # unet
        parser.add_argument("--train-unet", required=False, action='store_true', dest="train_unet", help="Whether the unet should be trained")
        parser.add_argument("--train-unet-epochs", type=int, required=False, default=2 ** 30, dest="train_unet_epochs", help="Number of epochs to train the unet for")
        parser.add_argument("--unet-learning-rate", type=float, required=False, default=None, dest="unet_learning_rate", help="Learning rate for the unet")
        parser.add_argument("--offset-noise-weight", type=float, required=False, default=0.0, dest="offset_noise_weight", help="The weight for offset noise prediction")
        parser.add_argument("--perturbation-noise-weight", type=float, required=False, default=0.0, dest="perturbation_noise_weight", help="The weight for perturbation noise")
        parser.add_argument("--rescale-noise-scheduler-to-zero-terminal-snr", required=False, action='store_true', dest="rescale_noise_scheduler_to_zero_terminal_snr", help="Rescales the noise sceduler to have a zero terminal signal to noise ratio, this also sets the model to v-prediction mode")
        parser.add_argument("--force-v-prediction", required=False, action='store_true', dest="force_v_prediction", help="Forces the training to use v-prediction")
        parser.add_argument("--force-epsilon-prediction", required=False, action='store_true', dest="force_epsilon_prediction", help="Forces the training to use epsilon-prediction")
        parser.add_argument("--max-noising-strength", type=float, required=False, default=1.0, dest="max_noising_strength", help="The max noising strength for training. Useful to prevent overfitting")
        parser.add_argument("--unet-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="unet_weight_dtype", help="The data type to use for unet weights during training", choices=list(DataType))

        # prior
        parser.add_argument("--train-prior", required=False, action='store_true', dest="train_prior", help="Whether the unet should be trained")
        parser.add_argument("--train-prior-epochs", type=int, required=False, default=2 ** 30, dest="train_prior_epochs", help="Number of epochs to train the unet for")
        parser.add_argument("--prior-learning-rate", type=float, required=False, default=None, dest="prior_learning_rate", help="Learning rate for the unet")
        parser.add_argument("--prior-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="prior_weight_dtype", help="The data type to use for unet weights during training", choices=list(DataType))

        # text encoder
        parser.add_argument("--train-text-encoder", required=False, action='store_true', dest="train_text_encoder", help="Whether the text encoder should be trained")
        parser.add_argument("--train-text-encoder-epochs", type=int, required=False, default=2 ** 30, dest="train_text_encoder_epochs", help="Number of epochs to train the text encoder for")
        parser.add_argument("--text-encoder-learning-rate", type=float, required=False, default=None, dest="text_encoder_learning_rate", help="Learning rate for the text encoder")
        parser.add_argument("--text-encoder-layer-skip", type=int, required=False, default=0, dest="text_encoder_layer_skip", help="Skip last layers of the text encoder")
        parser.add_argument("--text-encoder-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="text_encoder_weight_dtype", help="The data type to use for text encoder weights during training", choices=list(DataType))

        # text encoder 2
        parser.add_argument("--train-text-encoder-2", required=False, action='store_true', dest="train_text_encoder_2", help="Whether the text encoder 2 should be trained")
        parser.add_argument("--train-text-encoder-2-epochs", type=int, required=False, default=2 ** 30, dest="train_text_encoder_2_epochs", help="Number of epochs to train the text encoder 2 for")
        parser.add_argument("--text-encoder-2-learning-rate", type=float, required=False, default=None, dest="text_encoder_2_learning_rate", help="Learning rate for the text encoder 2")
        parser.add_argument("--text-encoder-2-layer-skip", type=int, required=False, default=0, dest="text_encoder_2_layer_skip", help="Skip last layers of the text encoder 2")
        parser.add_argument("--text-encoder-2-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="text_encoder_2_weight_dtype", help="The data type to use for text encoder 2 weights during training", choices=list(DataType))

        # vae
        parser.add_argument("--vae-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="vae_weight_dtype", help="The data type to use for vae weights during training", choices=list(DataType))

        # effnet encoder
        parser.add_argument("--effnet-encoder-model-name", type=str, required=False, dest="effnet_encoder_model_name", default="", help="The effnet encoder model to start training from")
        parser.add_argument("--effnet-encoder-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="effnet_encoder_weight_dtype", help="The data type to use for effnet encoder weights during training", choices=list(DataType))

        # decoder
        parser.add_argument("--decoder-model-name", type=str, required=True, dest="decoder_model_name", default="", help="The decoder model to start training from")
        parser.add_argument("--decoder-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="decoder_weight_dtype", help="The data type to use for decoder weights during training", choices=list(DataType))

        # decoder text encoder
        parser.add_argument("--decoder-text-encoder-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="decoder_text_encoder_weight_dtype", help="The data type to use for decoder text encoder weights during training", choices=list(DataType))

        # decoder vqgan
        parser.add_argument("--decoder-vqgan-weight-dtype", type=DataType, required=False, default=DataType.NONE, dest="decoder_vqgan_weight_dtype", help="The data type to use for decoder vqgan weights during training", choices=list(DataType))

        # masked training
        parser.add_argument("--masked-training", required=False, action='store_true', dest="masked_training", help="Activates masked training to let the model focus on certain parts of the training sample")
        parser.add_argument("--unmasked-probability", type=float, required=False, default=0.0, dest="unmasked_probability", help="If masked training is active, defines the number of steps to train on unmasked samples")
        parser.add_argument("--unmasked-weight", type=float, required=False, default=0.0, dest="unmasked_weight", help="If masked training is active, defines the loss weight of the unmasked parts of the image")
        parser.add_argument("--normalize-masked-area-loss", required=False, action='store_true', dest="normalize_masked_area_loss", help="If masked training is active, normalizes the loss based on the masked region for each sample")

        # embedding
        parser.add_argument("--embedding-model-name", type=str, required=False, action="append", default=[], dest="embedding_model_names", help="The embedding to start training from")
        parser.add_argument("--token-count", type=int, required=False, default=1, dest="token_count", help="The number of tokens to train")
        parser.add_argument("--initial-embedding-text", type=str, required=False, default="*", dest="initial_embedding_text", help="The text to initialize new embeddings")
        parser.add_argument("--embedding-weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="embedding_weight_dtype", help="The data type to use for training the Embedding", choices=list(DataType))

        # lora
        parser.add_argument("--lora-model-name", type=str, required=False, default=None, dest="lora_model_name", help="The LoRA to start training from")
        parser.add_argument("--lora-rank", type=int, required=False, default=1, dest="lora_rank", help="The rank parameter used when initializing new LoRA networks")
        parser.add_argument("--lora-alpha", type=float, required=False, default=1.0, dest="lora_alpha", help="The alpha parameter used when initializing new LoRA networks")
        parser.add_argument("--lora-weight-dtype", type=DataType, required=False, default=DataType.FLOAT_32, dest="lora_weight_dtype", help="The data type to use for training the LoRA", choices=list(DataType))

        # optimizer settings
        parser.add_argument("--optimizer-adam-w-mode", type=nullable_bool, default=None, dest="optimizer_adam_w_mode", help='Whether to use weight decay correction for Adam optimizer.')
        parser.add_argument("--optimizer-alpha", type=float, default=None, dest="optimizer_alpha", help='Smoothing parameter for RMSprop and others.')
        parser.add_argument("--optimizer-amsgrad", type=nullable_bool, default=None, dest="optimizer_amsgrad", help='Whether to use the AMSGrad variant for Adam.')
        parser.add_argument("--optimizer-beta1", type=float, default=None, dest="optimizer_beta1", help='optimizer_momentum term.')
        parser.add_argument("--optimizer-beta2", type=float, default=None, dest="optimizer_beta2", help='Coefficients for computing running averages of gradient.')
        parser.add_argument("--optimizer-beta3", type=float, default=None, dest="optimizer_beta3", help='Coefficient for computing the Prodigy stepsize.')
        parser.add_argument("--optimizer-bias-correction", type=nullable_bool, default=None, dest="optimizer_bias_correction", help='Whether to use bias correction in optimization algorithms like Adam.')
        parser.add_argument("--optimizer-block-wise", type=nullable_bool, default=None, dest="optimizer_block_wise", help='Whether to perform block-wise model update.')
        parser.add_argument("--optimizer-capturable", type=nullable_bool, default=None, dest="optimizer_capturable", help='Whether some property of the optimizer can be captured.')
        parser.add_argument("--optimizer-centered", type=nullable_bool, default=None, dest="optimizer_centered", help='Whether to center the gradient before scaling. Great for stabilizing the training process.')
        parser.add_argument("--optimizer-clip-threshold", type=float, default=None, dest="optimizer_clip_threshold", help='Clipping value for gradients.')
        parser.add_argument("--optimizer-d0", type=float, default=None, dest="optimizer_d0", help='Initial D estimate for D-adaptation.')
        parser.add_argument("--optimizer-d-coef", type=float, default=None, dest="optimizer_d_coef", help='Coefficient in the expression for the estimate of d.')
        parser.add_argument("--optimizer-dampening", type=float, default=None, dest="optimizer_dampening", help='Dampening for optimizer_momentum.')
        parser.add_argument("--optimizer-decay-rate", type=float, default=None, dest="optimizer_decay_rate", help='Rate of decay for moment estimation.')
        parser.add_argument("--optimizer-decouple", type=nullable_bool, default=None, dest="optimizer_decouple", help='Use AdamW style optimizer_decoupled weight decay.')
        parser.add_argument("--optimizer-differentiable", type=nullable_bool, default=None, dest="optimizer_differentiable", help='Whether the optimization function is optimizer_differentiable.')
        parser.add_argument("--optimizer-eps", type=float, default=None, dest="optimizer_eps", help='A small value to prevent division by zero.')
        parser.add_argument("--optimizer-eps2", type=float, default=None, dest="optimizer_eps2", help='A small value to prevent division by zero.')
        parser.add_argument("--optimizer-foreach", type=nullable_bool, default=None, dest="optimizer_foreach", help='If true, apply the optimizer to each parameter independently.')
        parser.add_argument("--optimizer-fsdp-in-use", type=nullable_bool, default=None, dest="optimizer_fsdp_in_use", help='Flag for using sharded parameters.')
        parser.add_argument("--optimizer-fused", type=nullable_bool, default=None, dest="optimizer_fused", help='Whether to use a optimizer_fused implementation if available.')
        parser.add_argument("--optimizer-growth-rate", type=float, default=None, dest="optimizer_growth_rate", help='Limit for D estimate growth rate.')
        parser.add_argument("--optimizer-initial-accumulator-value", type=float, default=None, dest="optimizer_initial_accumulator_value", help='Initial value for Adagrad optimizer.')
        parser.add_argument("--optimizer-is-paged", type=nullable_bool, default=None, dest="optimizer_is_paged", help='Whether the optimizer\'s internal state should be paged to CPU.')
        parser.add_argument("--optimizer-log-every", type=int, default=None, dest="optimizer_log_every", help='Intervals at which logging should occur.')
        parser.add_argument("--optimizer-lr-decay", type=float, default=None, dest="optimizer_lr_decay", help='Rate at which learning rate decreases.')
        parser.add_argument("--optimizer-max-unorm", type=float, default=None, dest="optimizer_max_unorm", help='Maximum value for gradient clipping by norms.')
        parser.add_argument("--optimizer-maximize", type=nullable_bool, default=None, dest="optimizer_maximize", help='Whether to optimizer_maximize the optimization function.')
        parser.add_argument("--optimizer-min-8bit-size", type=int, default=None, dest="optimizer_min_8bit_size", help='Minimum tensor size for 8-bit quantization.')
        parser.add_argument("--optimizer-momentum", type=float, default=None, dest="optimizer_momentum", help='Factor to accelerate SGD in relevant direction.')
        parser.add_argument("--optimizer-nesterov", type=nullable_bool, default=None, dest="optimizer_nesterov", help='Whether to enable Nesterov optimizer_momentum.')
        parser.add_argument("--optimizer-no-prox", type=nullable_bool, default=None, dest="optimizer_no_prox", help='Whether to use proximity updates or not.')
        parser.add_argument("--optimizer-optim-bits", type=int, default=None, dest="optimizer_optim_bits", help='Number of bits used for optimization.')
        parser.add_argument("--optimizer-percentile-clipping", type=float, default=None, dest="optimizer_percentile_clipping", help='Gradient clipping based on percentile values.')
        parser.add_argument("--optimizer-relative-step", type=nullable_bool, default=None, dest="optimizer_relative_step", help='Whether to use a relative step size.')
        parser.add_argument("--optimizer-safeguard-warmup", type=nullable_bool, default=None, dest="optimizer_safeguard_warmup", help='Avoid issues during warm-up stage.')
        parser.add_argument("--optimizer-scale-parameter", type=nullable_bool, default=None, dest="optimizer_scale_parameter", help='Whether to scale the parameter or not.')
        parser.add_argument("--optimizer-use-bias-correction", type=nullable_bool, default=None, dest="optimizer_use_bias_correction", help='Turn on Adam\'s bias correction.')
        parser.add_argument("--optimizer-use-triton", type=nullable_bool, default=None, dest="optimizer_use_triton", help='Whether Triton optimization should be used.')
        parser.add_argument("--optimizer-warmup-init", type=nullable_bool, default=None, dest="optimizer_warmup_init", help='Whether to warm-up the optimizer initialization.')
        parser.add_argument("--optimizer-weight-decay", type=float, default=None, dest="optimizer_weight_decay", help='Regularization to prevent overfitting.')

        # sample settings
        parser.add_argument("--sample-definition-file-name", type=str, required=True, dest="sample_definition_file_name", help="The json file containing the sample definition")
        parser.add_argument("--sample-after", type=float, required=True, dest="sample_after", help="The interval to sample")
        parser.add_argument("--sample-after-unit", type=TimeUnit, required=True, dest="sample_after_unit", help="The unit applied to the sample-after option")
        parser.add_argument("--sample-image-format", type=ImageFormat, required=False, default=ImageFormat.JPG, dest="sample_image_format", help="The file format used when saving samples", choices=list(ImageFormat))
        parser.add_argument("--samples-to-tensorboard", required=False, action='store_true', dest="samples_to_tensorboard", help="Whether to include sample images in the Tensorboard output.")
        parser.add_argument("--non-ema-sampling", required=False, action='store_true', dest="non_ema_sampling", help="Whether to include non-ema sampling when using ema.")

        # backup settings
        parser.add_argument("--backup-after", type=float, required=True, dest="backup_after", help="The interval for backups")
        parser.add_argument("--backup-after-unit", type=TimeUnit, required=True, dest="backup_after_unit", help="The unit applied to the backup-after option")
        parser.add_argument("--rolling-backup", required=False, action='store_true', dest="rolling_backup", help="Enable rolling backups")
        parser.add_argument("--rolling-backup-count", type=int, required=False, default=3, dest="rolling_backup_count", help="The number of backups to keep if rolling backups are enabled")
        parser.add_argument("--backup-before-save", required=False, action='store_true', dest="backup_before_save", help="Create a backup before saving the final model")
        parser.add_argument("--save-after", type=float, required=False, default=0, dest="save_after", help="The interval for backups")
        parser.add_argument("--save-after-unit", type=TimeUnit, required=False, default=TimeUnit.NEVER, dest="save_after_unit", help="The unit applied to the backup-after option")

        # @formatter:on

        args = TrainArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values() -> 'TrainArgs':
        data = []

        # name, default value, data type, nullable

        # general settings
        data.append(("training_method", TrainingMethod.FINE_TUNE, TrainingMethod, False))
        data.append(("model_type", ModelType.STABLE_DIFFUSION_15, ModelType, False))
        data.append(("debug_mode", False, bool, False))
        data.append(("debug_dir", "debug", str, False))
        data.append(("workspace_dir", "workspace/run", str, False))
        data.append(("cache_dir", "workspace-cache/run", str, False))
        data.append(("tensorboard", True, bool, False))
        data.append(("continue_last_backup", False, bool, False))

        # model settings
        data.append(("base_model_name", "runwayml/stable-diffusion-v1-5", str, False))
        data.append(("weight_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("output_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("output_model_format", ModelFormat.SAFETENSORS, ModelFormat, False))
        data.append(("output_model_destination", "models/model.safetensors", str, False))
        data.append(("gradient_checkpointing", True, bool, False))

        # data settings
        data.append(("concept_file_name", "training_concepts/concepts.json", str, False))
        data.append(("circular_mask_generation", False, bool, False))
        data.append(("random_rotate_and_crop", False, bool, False))
        data.append(("aspect_ratio_bucketing", True, bool, False))
        data.append(("latent_caching", True, bool, False))
        data.append(("latent_caching_epochs", 1, int, False))
        data.append(("clear_cache_before_training", True, bool, False))

        # training settings
        data.append(("learning_rate_scheduler", LearningRateScheduler.CONSTANT, LearningRateScheduler, False))
        data.append(("learning_rate", 3e-6, float, False))
        data.append(("learning_rate_warmup_steps", 200, int, False))
        data.append(("learning_rate_cycles", 1, int, False))
        data.append(("epochs", 100, int, False))
        data.append(("batch_size", 1, int, False))
        data.append(("gradient_accumulation_steps", 1, int, False))
        data.append(("ema", EMAMode.OFF, EMAMode, False))
        data.append(("ema_decay", 0.999, float, False))
        data.append(("ema_update_step_interval", 5, int, False))
        data.append(("train_device", "cuda", str, False))
        data.append(("temp_device", "cpu", str, False))
        data.append(("train_dtype", DataType.FLOAT_16, DataType, False))
        data.append(("only_cache", False, bool, False))
        data.append(("resolution", "512", str, False))
        data.append(("attention_mechanism", AttentionMechanism.XFORMERS, AttentionMechanism, False))
        data.append(("align_prop", False, bool, False))
        data.append(("align_prop_probability", 0.1, float, False))
        data.append(("align_prop_loss", AlignPropLoss.AESTHETIC, AlignPropLoss, False))
        data.append(("align_prop_weight", 0.01, float, False))
        data.append(("align_prop_steps", 20, int, False))
        data.append(("align_prop_truncate_steps", 0.5, float, False))
        data.append(("align_prop_cfg_scale", 7.0, float, False))
        data.append(("mse_strength", 1.0, float, False))
        data.append(("mae_strength", 0.0, float, False))
        data.append(("loss_scaler", LossScaler.NONE, LossScaler, False))
        data.append(("learning_rate_scaler", LearningRateScaler.NONE, LearningRateScaler, False))

        # unet
        data.append(("train_unet", True, bool, False))
        data.append(("train_unet_epochs", 10000, int, False))
        data.append(("unet_learning_rate", None, float, True))
        data.append(("offset_noise_weight", 0.0, float, False))
        data.append(("perturbation_noise_weight", 0.0, float, False))
        data.append(("rescale_noise_scheduler_to_zero_terminal_snr", False, bool, False))
        data.append(("force_v_prediction", False, bool, False))
        data.append(("force_epsilon_prediction", False, bool, False))
        data.append(("max_noising_strength", 1.0, float, False))
        data.append(("unet_weight_dtype", DataType.NONE, DataType, False))

        # prior
        data.append(("train_prior", True, bool, False))
        data.append(("train_prior_epochs", 10000, int, False))
        data.append(("prior_learning_rate", None, float, True))
        data.append(("prior_weight_dtype", DataType.NONE, DataType, False))

        # text encoder
        data.append(("train_text_encoder", True, bool, False))
        data.append(("train_text_encoder_epochs", 30, int, False))
        data.append(("text_encoder_learning_rate", None, float, True))
        data.append(("text_encoder_layer_skip", 0, int, False))
        data.append(("text_encoder_weight_dtype", DataType.NONE, DataType, False))

        # text encoder 2
        data.append(("train_text_encoder_2", True, bool, False))
        data.append(("train_text_encoder_2_epochs", 30, int, False))
        data.append(("text_encoder_2_learning_rate", None, float, True))
        data.append(("text_encoder_2_layer_skip", 0, int, False))
        data.append(("text_encoder_2_weight_dtype", DataType.NONE, DataType, False))

        # vae
        data.append(("vae_weight_dtype", DataType.FLOAT_32, DataType, False))

        # effnet encoder
        data.append(("effnet_encoder_model_name", "", str, False))
        data.append(("effnet_encoder_weight_dtype", DataType.NONE, DataType, False))

        # decoder
        data.append(("decoder_model_name", "", str, False))
        data.append(("decoder_weight_dtype", DataType.NONE, DataType, False))

        # decoder text encoder
        data.append(("decoder_text_encoder_weight_dtype", DataType.NONE, DataType, False))

        # decoder vqgan
        data.append(("decoder_vqgan_weight_dtype", DataType.NONE, DataType, False))

        # masked training
        data.append(("masked_training", False, bool, False))
        data.append(("unmasked_probability", 0.1, float, False))
        data.append(("unmasked_weight", 0.1, float, False))
        data.append(("normalize_masked_area_loss", False, bool, False))

        # embedding
        data.append(("embedding_model_names", [], list[str], False))
        data.append(("token_count", 1, int, False))
        data.append(("initial_embedding_text", "*", str, False))
        data.append(("embedding_weight_dtype", DataType.FLOAT_32, DataType, False))

        # lora
        data.append(("lora_model_name", "", str, False))
        data.append(("lora_rank", 16, int, False))
        data.append(("lora_alpha", 1.0, float, False))
        data.append(("lora_weight_dtype", DataType.FLOAT_32, DataType, False))

        # optimizer settings
        data.append(("optimizer", Optimizer.ADAMW, Optimizer, False))
        data.append(("optimizer_adam_w_mode", None, bool, True))
        data.append(("optimizer_alpha", None, float, True))
        data.append(("optimizer_amsgrad", None, bool, True))
        data.append(("optimizer_beta1", None, float, True))
        data.append(("optimizer_beta2", None, float, True))
        data.append(("optimizer_beta3", None, float, True))
        data.append(("optimizer_bias_correction", None, bool, True))
        data.append(("optimizer_block_wise", None, bool, True))
        data.append(("optimizer_capturable", None, bool, True))
        data.append(("optimizer_centered", None, bool, True))
        data.append(("optimizer_clip_threshold", None, float, True))
        data.append(("optimizer_d0", None, float, True))
        data.append(("optimizer_d_coef", None, float, True))
        data.append(("optimizer_dampening", None, float, True))
        data.append(("optimizer_decay_rate", None, float, True))
        data.append(("optimizer_decouple", None, bool, True))
        data.append(("optimizer_differentiable", None, bool, True))
        data.append(("optimizer_eps", None, float, True))
        data.append(("optimizer_eps2", None, float, True))
        data.append(("optimizer_foreach", None, bool, True))  # Disabled, because it uses too much VRAM
        data.append(("optimizer_fsdp_in_use", None, bool, True))
        data.append(("optimizer_fused", None, bool, True))
        data.append(("optimizer_growth_rate", None, float, True))
        data.append(("optimizer_initial_accumulator_value", None, int, True))
        data.append(("optimizer_is_paged", None, bool, True))
        data.append(("optimizer_log_every", None, int, True))
        data.append(("optimizer_lr_decay", None, float, True))
        data.append(("optimizer_max_unorm", None, float, True))
        data.append(("optimizer_maximize", None, bool, True))
        data.append(("optimizer_min_8bit_size", None, int, True))
        data.append(("optimizer_momentum", None, float, True))
        data.append(("optimizer_nesterov", None, bool, True))
        data.append(("optimizer_no_prox", None, bool, True))
        data.append(("optimizer_optim_bits", None, int, True))
        data.append(("optimizer_percentile_clipping", None, float, True))
        data.append(("optimizer_relative_step", None, bool, True))
        data.append(("optimizer_safeguard_warmup", None, bool, True))
        data.append(("optimizer_scale_parameter", None, bool, True))
        data.append(("optimizer_use_bias_correction", None, bool, True))
        data.append(("optimizer_use_triton", None, bool, True))
        data.append(("optimizer_warmup_init", None, bool, True))
        data.append(("optimizer_weight_decay", None, float, True))

        # sample settings
        data.append(("sample_definition_file_name", "training_samples/samples.json", str, False))
        data.append(("sample_after", 10, int, False))
        data.append(("sample_after_unit", TimeUnit.MINUTE, TimeUnit, False))
        data.append(("sample_image_format", ImageFormat.JPG, ImageFormat, False))
        data.append(("samples_to_tensorboard", True, bool, False))
        data.append(("non_ema_sampling", True, bool, False))

        # backup settings
        data.append(("backup_after", 30, int, False))
        data.append(("backup_after_unit", TimeUnit.MINUTE, TimeUnit, False))
        data.append(("rolling_backup", False, bool, False))
        data.append(("rolling_backup_count", 3, int, False))
        data.append(("backup_before_save", True, bool, False))
        data.append(("save_after", 0, int, False))
        data.append(("save_after_unit", TimeUnit.NEVER, TimeUnit, False))

        return TrainArgs(data)
