import json
import os
import uuid
from copy import deepcopy
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.CloudConfig import CloudConfig
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.ModelNames import EmbeddingName, ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.torch_util import default_device


class TrainOptimizerConfig(BaseConfig):
    optimizer: Optimizer
    adam_w_mode: bool
    alpha: float
    amsgrad: bool
    beta1: float
    beta2: float
    beta3: float
    bias_correction: bool
    block_wise: bool
    capturable: bool
    centered: bool
    clip_threshold: float
    d0: float
    d_coef: float
    dampening: float
    decay_rate: float
    decouple: bool
    differentiable: bool
    eps: float
    eps2: float
    foreach: bool
    fsdp_in_use: bool
    fused: bool
    fused_back_pass: bool
    growth_rate: float
    initial_accumulator_value: int
    initial_accumulator: float
    is_paged: bool
    log_every: int
    lr_decay: float
    max_unorm: float
    maximize: bool
    min_8bit_size: int
    quant_block_size: int
    momentum: float
    nesterov: bool
    no_prox: bool
    optim_bits: int
    percentile_clipping: int
    r: float
    relative_step: bool
    safeguard_warmup: bool
    scale_parameter: bool
    stochastic_rounding: bool
    use_bias_correction: bool
    use_triton: bool
    warmup_init: bool
    weight_decay: float
    weight_lr_power: float
    decoupled_decay: bool
    fixed_decay: bool
    weight_decouple: bool
    rectify: bool
    degenerated_to_sgd: bool
    k: int
    xi: float
    n_sma_threshold: int
    ams_bound: bool
    r: float
    adanorm: bool
    adam_debias: bool
    slice_p: int
    cautious: bool
    weight_decay_by_lr: True
    prodigy_steps: 0
    use_speed: False
    split_groups: True
    split_groups_mean: True
    factored: True
    factored_fp32: True
    use_stableadamw: True
    use_cautious: False
    use_grams: False
    use_adopt: False
    d_limiter: True
    use_schedulefree: True
    use_orthograd: False
    nnmf_factor: False
    orthogonal_gradient: False
    use_atan2: False
    use_AdEMAMix: False
    beta3_ema: float
    alpha_grad: float
    beta1_warmup: int
    min_beta1: float
    Simplified_AdEMAMix: False
    cautious_mask: False
    grams_moment: False
    kourkoutas_beta: False
    k_warmup_steps: int
    schedulefree_c: float
    ns_steps: int
    MuonWithAuxAdam: False
    muon_hidden_layers: str
    muon_adam_regex: False
    muon_adam_lr: float
    muon_te1_adam_lr: float
    muon_te2_adam_lr: float
    muon_adam_config: dict
    rms_rescaling: True
    normuon_variant: False
    beta2_normuon: float
    normuon_eps: float
    low_rank_ortho: False
    ortho_rank: int
    accelerated_ns: False
    cautious_wd: False
    approx_mars: False
    kappa_p: float
    auto_kappa_p: False

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("optimizer", Optimizer.ADAMW, Optimizer, False))
        data.append(("adam_w_mode", False, bool, False))
        data.append(("alpha", None, float, True))
        data.append(("amsgrad", False, bool, False))
        data.append(("beta1", None, float, True))
        data.append(("beta2", None, float, True))
        data.append(("beta3", None, float, True))
        data.append(("bias_correction", False, bool, False))
        data.append(("block_wise", False, bool, False))
        data.append(("capturable", False, bool, False))
        data.append(("centered", False, bool, False))
        data.append(("clip_threshold", None, float, True))
        data.append(("d0", None, float, True))
        data.append(("d_coef", None, float, True))
        data.append(("dampening", None, float, True))
        data.append(("decay_rate", None, float, True))
        data.append(("decouple", False, bool, False))
        data.append(("differentiable", False, bool, False))
        data.append(("eps", None, float, True))
        data.append(("eps2", None, float, True))
        data.append(("foreach", False, bool, True))  # Disabled, because it uses too much VRAM
        data.append(("fsdp_in_use", False, bool, False))
        data.append(("fused", False, bool, False))
        data.append(("fused_back_pass", False, bool, False))
        data.append(("growth_rate", None, float, True))
        data.append(("initial_accumulator_value", None, int, True))
        data.append(("initial_accumulator", None, float, True))
        data.append(("is_paged", False, bool, False))
        data.append(("log_every", None, int, True))
        data.append(("lr_decay", None, float, True))
        data.append(("max_unorm", None, float, True))
        data.append(("maximize", False, bool, False))
        data.append(("min_8bit_size", None, int, True))
        data.append(("quant_block_size", None, int, True))
        data.append(("momentum", None, float, True))
        data.append(("nesterov", False, bool, False))
        data.append(("no_prox", False, bool, False))
        data.append(("optim_bits", None, int, True))
        data.append(("percentile_clipping", None, int, True))
        data.append(("r", None, float, True))
        data.append(("relative_step", False, bool, False))
        data.append(("safeguard_warmup", False, bool, False))
        data.append(("scale_parameter", False, bool, False))
        data.append(("stochastic_rounding", True, bool, False))
        data.append(("use_bias_correction", False, bool, False))
        data.append(("use_triton", False, bool, False))
        data.append(("warmup_init", False, bool, False))
        data.append(("weight_decay", None, float, True))
        data.append(("weight_lr_power", None, float, True))
        data.append(("decoupled_decay", False, bool, False))
        data.append(("fixed_decay", False, bool, False))
        data.append(("rectify", False, bool, False))
        data.append(("degenerated_to_sgd", False, bool, False))
        data.append(("k", None, int, True))
        data.append(("xi", None, float, True))
        data.append(("n_sma_threshold", None, int, True))
        data.append(("ams_bound", False, bool, False))
        data.append(("r", None, float, True))
        data.append(("adanorm", False, bool, False))
        data.append(("adam_debias", False, bool, False))
        data.append(("slice_p", None, int, True))
        data.append(("cautious", False, bool, False))
        data.append(("weight_decay_by_lr", True, bool, False))
        data.append(("prodigy_steps", None, int, True))
        data.append(("use_speed", False, bool, False))
        data.append(("split_groups", True, bool, False))
        data.append(("split_groups_mean", True, bool, False))
        data.append(("factored", True, bool, False))
        data.append(("factored_fp32", True, bool, False))
        data.append(("use_stableadamw", True, bool, False))
        data.append(("use_cautious", False, bool, False))
        data.append(("use_grams", False, bool, False))
        data.append(("use_adopt", False, bool, False))
        data.append(("d_limiter", True, bool, True))
        data.append(("use_schedulefree", True, bool, True))
        data.append(("use_orthograd", False, bool, False))
        data.append(("nnmf_factor", False, bool, False))
        data.append(("orthogonal_gradient", False, bool, False))
        data.append(("use_atan2", False, bool, False))
        data.append(("use_AdEMAMix", False, bool, False))
        data.append(("beta3_ema", None, float, True))
        data.append(("alpha_grad", None, float, True))
        data.append(("beta1_warmup", None, int, True))
        data.append(("min_beta1", None, float, True))
        data.append(("Simplified_AdEMAMix", False, bool, False))
        data.append(("cautious_mask", False, bool, False))
        data.append(("grams_moment", False, bool, False))
        data.append(("kourkoutas_beta", False, bool, False))
        data.append(("k_warmup_steps", None, int, True))
        data.append(("schedulefree_c", None, float, True))
        data.append(("ns_steps", None, int, True))
        data.append(("MuonWithAuxAdam", False, bool, False))
        data.append(("muon_hidden_layers", None, str, True))
        data.append(("muon_adam_regex", False, bool, False))
        data.append(("muon_adam_lr", None, float, True))
        data.append(("muon_te1_adam_lr", None, float, True))
        data.append(("muon_te2_adam_lr", None, float, True))
        data.append(("muon_adam_config", None, dict, True))
        data.append(("rms_rescaling", True, bool, True))
        data.append(("normuon_variant", False, bool, False))
        data.append(("beta2_normuon", None, float, True))
        data.append(("normuon_eps", None, float, True))
        data.append(("low_rank_ortho", False, bool, False))
        data.append(("ortho_rank", None, int, True))
        data.append(("accelerated_ns", False, bool, False))
        data.append(("cautious_wd", False, bool, False))
        data.append(("approx_mars", False, bool, False))
        data.append(("kappa_p", None, float, True))
        data.append(("auto_kappa_p", False, bool, False))

        return TrainOptimizerConfig(data)


class TrainModelPartConfig(BaseConfig):
    model_name: str
    include: bool
    train: bool
    stop_training_after: int
    stop_training_after_unit: TimeUnit
    learning_rate: float
    weight_dtype: DataType
    dropout_probability: float
    train_embedding: bool
    attention_mask: bool
    guidance_scale: float

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("model_name", "", str, False))
        data.append(("include", True, bool, False))
        data.append(("train", True, bool, False))
        data.append(("stop_training_after", None, int, True))
        data.append(("stop_training_after_unit", TimeUnit.NEVER, TimeUnit, False))
        data.append(("learning_rate", None, float, True))
        data.append(("weight_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("dropout_probability", 0.0, float, False))
        data.append(("train_embedding", True, bool, False))
        data.append(("attention_mask", False, bool, False))
        data.append(("guidance_scale", 1.0, float, False))

        return TrainModelPartConfig(data)


class TrainEmbeddingConfig(BaseConfig):
    uuid: str
    model_name: str
    placeholder: str
    train: bool
    stop_training_after: int
    stop_training_after_unit: TimeUnit
    token_count: int | None
    initial_embedding_text: str
    is_output_embedding: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("uuid", str(uuid.uuid4()), str, False))
        data.append(("model_name", "", str, False))
        data.append(("placeholder", "<embedding>", str, False))
        data.append(("train", True, bool, False))
        data.append(("stop_training_after", None, int, True))
        data.append(("stop_training_after_unit", TimeUnit.NEVER, TimeUnit, False))
        data.append(("token_count", 1, int, True))
        data.append(("initial_embedding_text", "*", str, False))
        data.append(("is_output_embedding", False, bool, False))

        return TrainEmbeddingConfig(data)

class QuantizationConfig(BaseConfig):
    layer_filter: str
    layer_filter_preset: str
    layer_filter_regex: bool
    svd_dtype: DataType
    svd_rank: int
    cache_dir: str

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("layer_filter", "", str, False))
        data.append(("layer_filter_preset", "full", str, False))
        data.append(("layer_filter_regex", False, bool, False))
        data.append(("svd_dtype", DataType.NONE, DataType, False))
        data.append(("svd_rank", 16, int, False))
        data.append(("cache_dir", None, str, True))
        return QuantizationConfig(data)

class TrainConfig(BaseConfig):
    training_method: TrainingMethod
    model_type: ModelType
    debug_mode: bool
    debug_dir: str
    workspace_dir: str
    cache_dir: str
    tensorboard: bool
    tensorboard_expose: bool
    tensorboard_always_on: bool
    tensorboard_port: str
    validation: bool
    validate_after: float
    validate_after_unit: TimeUnit
    continue_last_backup: bool
    include_train_config: ConfigPart

    # multi-GPU
    multi_gpu: bool
    device_indexes: str
    gradient_reduce_prevision: GradientReducePrecision
    fused_gradient_reduce: bool
    async_gradient_reduce: bool
    async_gradient_reduce_buffer: int

    # model settings
    base_model_name: str
    output_dtype: DataType
    output_model_format: ModelFormat
    output_model_destination: str
    gradient_checkpointing: GradientCheckpointingMethod
    enable_async_offloading: bool
    enable_activation_offloading: bool
    layer_offload_fraction: float
    force_circular_padding: bool
    compile: bool

    # data settings
    concept_file_name: str
    concepts: list[ConceptConfig]
    aspect_ratio_bucketing: bool
    latent_caching: bool
    clear_cache_before_training: bool

    # training settings
    learning_rate_scheduler: LearningRateScheduler
    custom_learning_rate_scheduler: str | None
    # Dict keys are literally called "key" and "value"; not a tuple because
    # of restrictions with ConfigList.
    scheduler_params: list[dict[str, str]]
    learning_rate: float
    learning_rate_warmup_steps: float
    learning_rate_cycles: float
    learning_rate_min_factor: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    ema: EMAMode
    ema_decay: float
    ema_update_step_interval: int
    dataloader_threads: int
    train_device: str
    temp_device: str
    train_dtype: DataType
    fallback_train_dtype: DataType
    enable_autocast_cache: bool
    only_cache: bool
    resolution: str
    frames: str
    mse_strength: float
    mae_strength: float
    log_cosh_strength: float
    huber_strength: float
    huber_delta: float
    vb_loss_strength: float
    loss_weight_fn: LossWeight
    loss_weight_strength: float
    dropout_probability: float
    loss_scaler: LossScaler
    learning_rate_scaler: LearningRateScaler
    clip_grad_norm: float

    #layer filter
    layer_filter: str  # comma-separated
    layer_filter_preset: str
    layer_filter_regex: bool

    # noise
    offset_noise_weight: float
    generalized_offset_noise: bool
    perturbation_noise_weight: float
    rescale_noise_scheduler_to_zero_terminal_snr: bool
    force_v_prediction: bool
    force_epsilon_prediction: bool
    timestep_distribution: TimestepDistribution
    min_noising_strength: float
    max_noising_strength: float

    noising_weight: float
    noising_bias: float

    timestep_shift: float
    dynamic_timestep_shifting: bool

    # unet
    unet: TrainModelPartConfig

    # prior
    prior: TrainModelPartConfig

    # transformer
    transformer: TrainModelPartConfig
    quantization: QuantizationConfig

    # text encoder
    text_encoder: TrainModelPartConfig
    text_encoder_layer_skip: int

    # text encoder 2
    text_encoder_2: TrainModelPartConfig
    text_encoder_2_layer_skip: int
    text_encoder_2_sequence_length: int

    # text encoder 3
    text_encoder_3: TrainModelPartConfig
    text_encoder_3_layer_skip: int

    # text encoder 4
    text_encoder_4: TrainModelPartConfig
    text_encoder_4_layer_skip: int

    # vae
    vae: TrainModelPartConfig

    # effnet encoder
    effnet_encoder: TrainModelPartConfig

    # decoder
    decoder: TrainModelPartConfig

    # decoder text encoder
    decoder_text_encoder: TrainModelPartConfig

    # decoder vqgan
    decoder_vqgan: TrainModelPartConfig

    # masked training
    masked_training: bool
    unmasked_probability: float
    unmasked_weight: float
    normalize_masked_area_loss: bool
    masked_prior_preservation_weight: float

    # custom conditioning image
    custom_conditioning_image: bool

    # embedding
    embedding_learning_rate: float
    preserve_embedding_norm: bool
    embedding: TrainEmbeddingConfig
    additional_embeddings: list[TrainEmbeddingConfig]
    embedding_weight_dtype: DataType

    # lora
    peft_type: PeftType
    lora_model_name: str
    lora_rank: int
    lora_alpha: float
    lora_decompose: bool
    lora_decompose_norm_epsilon: bool
    lora_decompose_output_axis: bool
    lora_weight_dtype: DataType
    bundle_additional_embeddings: bool

    # oft
    oft_block_size: int
    oft_coft: bool
    coft_eps: float
    oft_block_share: bool

    # optimizer
    optimizer: TrainOptimizerConfig
    optimizer_defaults: dict[str, TrainOptimizerConfig]

    # sample settings
    sample_definition_file_name: str
    samples: list[SampleConfig]
    sample_after: float
    sample_after_unit: TimeUnit
    sample_skip_first: int
    sample_image_format: ImageFormat
    sample_video_format: VideoFormat
    sample_audio_format: AudioFormat
    samples_to_tensorboard: bool
    non_ema_sampling: bool

    # cloud settings
    cloud: CloudConfig

    # backup settings
    backup_after: float
    backup_after_unit: TimeUnit
    rolling_backup: bool
    rolling_backup_count: int
    backup_before_save: bool
    save_every: int
    save_every_unit: TimeUnit
    save_skip_first: int
    save_filename_prefix: str

    # secrets - not saved into config file
    secrets: SecretsConfig

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(
            data,
            config_version=10,
            config_migrations={
                0: self.__migration_0,
                1: self.__migration_1,
                2: self.__migration_2,
                3: self.__migration_3,
                4: self.__migration_4,
                5: self.__migration_5,
                6: self.__migration_6,
                7: self.__migration_7,
                8: self.__migration_8,
                9: self.__migration_9,
            }
        )

    def __migration_0(self, data: dict) -> dict:
        optimizer_settings = {}
        migrated_data = {}
        for key, value in data.items():
            # move optimizer settings to sub object
            if key == 'optimizer':
                optimizer_settings['optimizer'] = value
            elif key.startswith('optimizer'):
                optimizer_settings[key.removeprefix('optimizer_')] = value
            else:
                migrated_data[key] = value

        if 'optimizer' in optimizer_settings:
            migrated_data['optimizer'] = optimizer_settings
            migrated_data['optimizer_defaults'] = {
                optimizer_settings['optimizer']: deepcopy(optimizer_settings)
            }

        return migrated_data

    def __migration_1(self, data: dict) -> dict:
        migrated_data = {
            "unet": {},
            "prior": {},
            "text_encoder": {},
            "text_encoder_2": {},
            "vae": {},
            "effnet_encoder": {},
            "decoder": {},
            "decoder_text_encoder": {},
            "decoder_vqgan": {},
            "embeddings": [{}],
        }

        for key, value in data.items():
            if key == "train_unet":
                migrated_data["unet"]["train"] = value
            elif key == "train_unet_epochs":
                migrated_data["unet"]["stop_training_after"] = value
                migrated_data["unet"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "unet_learning_rate":
                migrated_data["unet"]["learning_rate"] = value
            elif key == "unet_weight_dtype":
                migrated_data["unet"]["weight_dtype"] = value

            elif key == "train_prior":
                migrated_data["prior"]["train"] = value
            elif key == "prior_model_name":
                migrated_data["prior"]["model_name"] = value
            elif key == "train_prior_epochs":
                migrated_data["prior"]["stop_training_after"] = value
                migrated_data["prior"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "prior_learning_rate":
                migrated_data["prior"]["learning_rate"] = value
            elif key == "prior_weight_dtype":
                migrated_data["prior"]["weight_dtype"] = value

            elif key == "train_text_encoder":
                migrated_data["text_encoder"]["train"] = value
            elif key == "train_text_encoder_epochs":
                migrated_data["text_encoder"]["stop_training_after"] = value
                migrated_data["text_encoder"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "text_encoder_learning_rate":
                migrated_data["text_encoder"]["learning_rate"] = value
            elif key == "text_encoder_weight_dtype":
                migrated_data["text_encoder"]["weight_dtype"] = value

            elif key == "train_text_encoder_2":
                migrated_data["text_encoder_2"]["train"] = value
            elif key == "train_text_encoder_2_epochs":
                migrated_data["text_encoder_2"]["stop_training_after"] = value
                migrated_data["text_encoder_2"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "text_encoder_2_learning_rate":
                migrated_data["text_encoder_2"]["learning_rate"] = value
            elif key == "text_encoder_2_weight_dtype":
                migrated_data["text_encoder_2"]["weight_dtype"] = value

            elif key == "vae_model_name":
                migrated_data["vae"]["model_name"] = value
            elif key == "vae_weight_dtype":
                migrated_data["vae"]["weight_dtype"] = value

            elif key == "effnet_encoder_model_name":
                migrated_data["effnet_encoder"]["model_name"] = value
            elif key == "effnet_encoder_weight_dtype":
                migrated_data["effnet_encoder"]["weight_dtype"] = value

            elif key == "decoder_model_name":
                migrated_data["decoder"]["model_name"] = value
            elif key == "decoder_weight_dtype":
                migrated_data["decoder"]["weight_dtype"] = value

            elif key == "decoder_text_encoder_weight_dtype":
                migrated_data["decoder_text_encoder"]["weight_dtype"] = value

            elif key == "decoder_vqgan_weight_dtype":
                migrated_data["decoder_vqgan"]["weight_dtype"] = value

            elif key == "embedding_model_names" and len(value) > 0:
                migrated_data["embeddings"][0]["model_name"] = value[0]
            elif key == "token_count":
                migrated_data["embeddings"][0]["token_count"] = value
            elif key == "initial_embedding_text":
                migrated_data["embeddings"][0]["initial_embedding_text"] = value

            else:
                migrated_data[key] = value

        return migrated_data

    def __migration_2(self, data: dict) -> dict:
        migrated_data = data.copy()
        min_snr_gamma = migrated_data.pop("min_snr_gamma", 0.0)
        model_type = ModelType(migrated_data.get("model_type", ModelType.STABLE_DIFFUSION_15))
        if min_snr_gamma:
            migrated_data["loss_weight_fn"] = LossWeight.MIN_SNR_GAMMA
            migrated_data["loss_weight_strength"] = min_snr_gamma
        elif model_type.is_wuerstchen():
            migrated_data["loss_weight_fn"] = LossWeight.P2
            migrated_data["loss_weight_strength"] = 1.0

        return migrated_data

    def __migration_3(self, data: dict) -> dict:
        migrated_data = data.copy()

        noising_weight = migrated_data.pop("noising_weight", 0.0)
        noising_bias = migrated_data.pop("noising_bias", 0.5)

        if noising_weight != 0:
            migrated_data["timestep_distribution"] = TimestepDistribution.SIGMOID
            migrated_data["noising_weight"] = noising_weight
            migrated_data["noising_bias"] = noising_bias - 0.5
        else:
            migrated_data["timestep_distribution"] = TimestepDistribution.UNIFORM
            migrated_data["noising_weight"] = 0.0
            migrated_data["noising_bias"] = 0.0

        return migrated_data

    def __migration_4(self, data: dict) -> dict:
        migrated_data = data.copy()

        gradient_checkpointing = migrated_data.pop("gradient_checkpointing", True)

        if gradient_checkpointing:
            migrated_data["gradient_checkpointing"] = GradientCheckpointingMethod.ON
        else:
            migrated_data["gradient_checkpointing"] = GradientCheckpointingMethod.OFF

        return migrated_data

    def __migration_5(self, data: dict) -> dict:
        migrated_data = data.copy()

        if "save_after" in migrated_data:
            migrated_data["save_every"] = migrated_data.pop("save_after")
        if "save_after_unit" in migrated_data:
            migrated_data["save_every_unit"] = migrated_data.pop("save_after_unit")

        return migrated_data

    def __migration_6(self, data: dict) -> dict:
        migrated_data = data.copy()

        # None is not a valid value, but there was a bug that allowed it, so old config files can have it set to None:
        if (
            "lora_layer_preset" in migrated_data
            and migrated_data["lora_layer_preset"] is None
        ):
            migrated_data["lora_layer_preset"] = "full"

        return migrated_data

    def __migration_7(self, data: dict) -> dict:
        migrated_data = data.copy()

        if "lora_layers" in migrated_data:
            migrated_data["layer_filter"] = migrated_data.pop("lora_layers")
        if "lora_layer_preset" in migrated_data:
            migrated_data["layer_filter_preset"] = migrated_data.pop("lora_layer_preset")
        if "lora_layers_regex" in migrated_data:
            migrated_data["layer_filter_regex"] = migrated_data.pop("lora_layers_regex")

        return migrated_data

    def __migration_8(self, data: dict) -> dict:
        migrated_data = data.copy()

        if migrated_data["model_type"] != "STABLE_CASCADE_1" and migrated_data["model_type"] != "WUERSTCHEN_2":
            migrated_data["transformer"] = migrated_data["prior"]

        return migrated_data

    def __migration_9(self, data: dict) -> dict:
        migrated_data = data.copy()

        def replace_dtype(part: str):
            if part in migrated_data and migrated_data[part]["weight_dtype"] == "NONE":
                migrated_data[part]["weight_dtype"] = migrated_data["weight_dtype"]
        replace_dtype("unet")
        replace_dtype("prior")
        replace_dtype("transformer")
        replace_dtype("text_encoder")
        replace_dtype("text_encoder_2")
        replace_dtype("text_encoder_3")
        replace_dtype("text_encoder_4")
        replace_dtype("vae")
        replace_dtype("effnet_encoder")
        replace_dtype("decoder")
        replace_dtype("decoder_text_encoder")
        replace_dtype("decoder_vqgan")
        migrated_data.pop("weight_dtype")

        return migrated_data

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes(
            self.train_dtype,
            self.fallback_train_dtype,
            self.unet.weight_dtype,
            self.prior.weight_dtype,
            self.transformer.weight_dtype,
            self.text_encoder.weight_dtype,
            self.text_encoder_2.weight_dtype,
            self.text_encoder_3.weight_dtype,
            self.text_encoder_4.weight_dtype,
            self.vae.weight_dtype,
            self.effnet_encoder.weight_dtype,
            self.decoder.weight_dtype,
            self.decoder_text_encoder.weight_dtype,
            self.decoder_vqgan.weight_dtype,
            self.lora_weight_dtype,
            self.embedding_weight_dtype,
        )

    def model_names(self) -> ModelNames:
        return ModelNames(
            base_model=self.base_model_name,
            prior_model=self.prior.model_name,
            transformer_model=self.transformer.model_name,
            effnet_encoder_model=self.effnet_encoder.model_name,
            decoder_model=self.decoder.model_name,
            text_encoder_4=self.text_encoder_4.model_name,
            vae_model=self.vae.model_name,
            lora=self.lora_model_name,
            embedding=EmbeddingName(self.embedding.uuid, self.embedding.model_name) \
                if self.training_method == TrainingMethod.EMBEDDING else None,
            additional_embeddings=[EmbeddingName(embedding.uuid, embedding.model_name) for embedding in
                                   self.additional_embeddings],
            include_text_encoder=self.text_encoder.include,
            include_text_encoder_2=self.text_encoder_2.include,
            include_text_encoder_3=self.text_encoder_3.include,
            include_text_encoder_4=self.text_encoder_4.include,
        )

    def train_any_embedding(self) -> bool:
        return ((self.training_method == TrainingMethod.EMBEDDING) and not self.embedding.is_output_embedding) \
            or any((embedding.train and not embedding.is_output_embedding) for embedding in self.additional_embeddings)

    def train_any_output_embedding(self) -> bool:
        return ((self.training_method == TrainingMethod.EMBEDDING) and self.embedding.is_output_embedding) \
            or any((embedding.train and embedding.is_output_embedding) for embedding in self.additional_embeddings)

    def train_text_encoder_or_embedding(self) -> bool:
        return (self.text_encoder.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding())

    def train_text_encoder_2_or_embedding(self) -> bool:
        return (self.text_encoder_2.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder_2.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding())

    def train_text_encoder_3_or_embedding(self) -> bool:
        return (self.text_encoder_3.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder_3.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding())

    def train_text_encoder_4_or_embedding(self) -> bool:
        return (self.text_encoder_4.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder_4.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding())

    def all_embedding_configs(self):
        if self.training_method == TrainingMethod.EMBEDDING:
            return self.additional_embeddings + [self.embedding]
        else:
            return self.additional_embeddings

    def get_last_backup_path(self) -> str | None:
        backups_path = os.path.join(self.workspace_dir, "backup")
        if os.path.exists(backups_path):
            backup_paths = sorted(
                [path for path in os.listdir(backups_path) if
                 os.path.isdir(os.path.join(backups_path, path))],
                reverse=True,
            )

            if backup_paths:
                last_backup_path = backup_paths[0]
                return os.path.join(backups_path, last_backup_path)

        return None

    def to_settings_dict(self, secrets: bool) -> dict:
        config = TrainConfig.default_values().from_dict(self.to_dict())

        config.concepts = None
        config.samples = None

        config_dict = config.to_dict()
        if not secrets:
            config_dict.pop('secrets',None)
        return config_dict

    def to_pack_dict(self, secrets: bool) -> dict:
        config = TrainConfig.default_values().from_dict(self.to_dict())

        if config.concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = json.load(f)
                for i in range(len(concepts)):
                    concepts[i] = ConceptConfig.default_values().from_dict(concepts[i])
                config.concepts = concepts

        if config.samples is None:
            with open(config.sample_definition_file_name, 'r') as f:
                samples = json.load(f)
                for i in range(len(samples)):
                    samples[i] = SampleConfig.default_values().from_dict(samples[i])
                config.samples = samples

        config_dict = config.to_dict()
        if not secrets:
            config_dict.pop('secrets',None)
        return config_dict

    def to_unpacked_config(self) -> 'TrainConfig':
        config = TrainConfig.default_values().from_dict(self.to_dict())
        config.concepts = None
        config.samples = None
        return config

    @staticmethod
    def default_values() -> 'TrainConfig':
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
        data.append(("tensorboard_expose", False, bool, False))
        data.append(("tensorboard_always_on", False, bool, False))
        data.append(("tensorboard_port", 6006, int, False))
        data.append(("validation", False, bool, False))
        data.append(("validate_after", 1, int, False))
        data.append(("validate_after_unit", TimeUnit.EPOCH, TimeUnit, False))
        data.append(("continue_last_backup", False, bool, False))
        data.append(("include_train_config", ConfigPart.NONE, ConfigPart, False))

        #multi-GPU
        data.append(("multi_gpu", False, bool, False))
        data.append(("device_indexes", "", str, False))
        data.append(("gradient_reduce_precision", GradientReducePrecision.FLOAT_32_STOCHASTIC, GradientReducePrecision, False))
        data.append(("fused_gradient_reduce", True, bool, False))
        data.append(("async_gradient_reduce", True, bool, False))
        data.append(("async_gradient_reduce_buffer", 100, int, False))

        # model settings
        data.append(("base_model_name", "stable-diffusion-v1-5/stable-diffusion-v1-5", str, False))
        data.append(("output_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("output_model_format", ModelFormat.SAFETENSORS, ModelFormat, False))
        data.append(("output_model_destination", "models/model.safetensors", str, False))
        data.append(("gradient_checkpointing", GradientCheckpointingMethod.ON, GradientCheckpointingMethod, False))
        data.append(("enable_async_offloading", True, bool, False))
        data.append(("enable_activation_offloading", True, bool, False))
        data.append(("layer_offload_fraction", 0.0, float, False))
        data.append(("force_circular_padding", False, bool, False))
        data.append(("compile", False, bool, False))

        # data settings
        data.append(("concept_file_name", "training_concepts/concepts.json", str, False))
        data.append(("concepts", None, list[ConceptConfig], True))
        data.append(("aspect_ratio_bucketing", True, bool, False))
        data.append(("latent_caching", True, bool, False))
        data.append(("clear_cache_before_training", True, bool, False))

        # training settings
        data.append(("learning_rate_scheduler", LearningRateScheduler.CONSTANT, LearningRateScheduler, False))
        data.append(("custom_learning_rate_scheduler", None, str, True))
        data.append(("scheduler_params", [], list[dict[str, str]], True))
        data.append(("learning_rate", 3e-6, float, False))
        data.append(("learning_rate_warmup_steps", 200.0, float, False))
        data.append(("learning_rate_cycles", 1.0, float, False))
        data.append(("learning_rate_min_factor", 0.0, float, False))
        data.append(("epochs", 100, int, False))
        data.append(("batch_size", 1, int, False))
        data.append(("gradient_accumulation_steps", 1, int, False))
        data.append(("ema", EMAMode.OFF, EMAMode, False))
        data.append(("ema_decay", 0.999, float, False))
        data.append(("ema_update_step_interval", 5, int, False))
        data.append(("dataloader_threads", 2, int, False))
        data.append(("train_device", default_device.type, str, False))
        data.append(("temp_device", "cpu", str, False))
        data.append(("train_dtype", DataType.FLOAT_16, DataType, False))
        data.append(("fallback_train_dtype", DataType.BFLOAT_16, DataType, False))
        data.append(("enable_autocast_cache", True, bool, False))
        data.append(("only_cache", False, bool, False))
        data.append(("resolution", "512", str, False))
        data.append(("frames", "25", str, False))
        data.append(("mse_strength", 1.0, float, False))
        data.append(("mae_strength", 0.0, float, False))
        data.append(("log_cosh_strength", 0.0, float, False))
        data.append(("huber_strength", 0.0, float, False))
        data.append(("huber_delta", 1.0, float, False))
        data.append(("vb_loss_strength", 1.0, float, False))
        data.append(("loss_weight_fn", LossWeight.CONSTANT, LossWeight, False))
        data.append(("loss_weight_strength", 5.0, float, False))
        data.append(("dropout_probability", 0.0, float, False))
        data.append(("loss_scaler", LossScaler.NONE, LossScaler, False))
        data.append(("learning_rate_scaler", LearningRateScaler.NONE, LearningRateScaler, False))
        data.append(("clip_grad_norm", 1.0, float, True))

        # noise
        data.append(("offset_noise_weight", 0.0, float, False))
        data.append(("generalized_offset_noise", False, bool, False))
        data.append(("perturbation_noise_weight", 0.0, float, False))
        data.append(("rescale_noise_scheduler_to_zero_terminal_snr", False, bool, False))
        data.append(("force_v_prediction", False, bool, False))
        data.append(("force_epsilon_prediction", False, bool, False))
        data.append(("min_noising_strength", 0.0, float, False))
        data.append(("max_noising_strength", 1.0, float, False))
        data.append(("timestep_distribution", TimestepDistribution.UNIFORM, TimestepDistribution, False))
        data.append(("noising_weight", 0.0, float, False))
        data.append(("noising_bias", 0.0, float, False))
        data.append(("timestep_shift", 1.0, float, False))
        data.append(("dynamic_timestep_shifting", False, bool, False))


        # unet
        unet = TrainModelPartConfig.default_values()
        unet.train = True
        unet.stop_training_after = 0
        unet.learning_rate = None
        data.append(("unet", unet, TrainModelPartConfig, False))

        # prior
        prior = TrainModelPartConfig.default_values()
        prior.model_name = ""
        prior.train = True
        prior.stop_training_after = 0
        prior.learning_rate = None
        data.append(("prior", prior, TrainModelPartConfig, False))

        # transformer
        transformer = TrainModelPartConfig.default_values()
        transformer.model_name = ""
        transformer.train = True
        transformer.stop_training_after = 0
        transformer.learning_rate = None
        data.append(("transformer", transformer, TrainModelPartConfig, False))

        #quantization layer filter
        quantization = QuantizationConfig.default_values()
        data.append(("quantization", quantization, QuantizationConfig, False))

        # text encoder
        text_encoder = TrainModelPartConfig.default_values()
        text_encoder.train = True
        text_encoder.stop_training_after = 30
        text_encoder.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder.learning_rate = None
        data.append(("text_encoder", text_encoder, TrainModelPartConfig, False))
        data.append(("text_encoder_layer_skip", 0, int, False))

        # text encoder 2
        text_encoder_2 = TrainModelPartConfig.default_values()
        text_encoder_2.train = True
        text_encoder_2.stop_training_after = 30
        text_encoder_2.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder_2.learning_rate = None
        data.append(("text_encoder_2", text_encoder_2, TrainModelPartConfig, False))
        data.append(("text_encoder_2_layer_skip", 0, int, False))
        data.append(("text_encoder_2_sequence_length", 77, int, True))

        # text encoder 3
        text_encoder_3 = TrainModelPartConfig.default_values()
        text_encoder_3.train = True
        text_encoder_3.stop_training_after = 30
        text_encoder_3.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder_3.learning_rate = None
        data.append(("text_encoder_3", text_encoder_3, TrainModelPartConfig, False))
        data.append(("text_encoder_3_layer_skip", 0, int, False))

        # text encoder 4
        text_encoder_4 = TrainModelPartConfig.default_values()
        text_encoder_4.train = True
        text_encoder_4.stop_training_after = 30
        text_encoder_4.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder_4.learning_rate = None
        data.append(("text_encoder_4", text_encoder_4, TrainModelPartConfig, False))
        data.append(("text_encoder_4_layer_skip", 0, int, False))

        # vae
        vae = TrainModelPartConfig.default_values()
        vae.model_name = ""
        data.append(("vae", vae, TrainModelPartConfig, False))

        # effnet encoder
        effnet_encoder = TrainModelPartConfig.default_values()
        effnet_encoder.model_name = ""
        data.append(("effnet_encoder", effnet_encoder, TrainModelPartConfig, False))

        # decoder
        decoder = TrainModelPartConfig.default_values()
        decoder.model_name = ""
        data.append(("decoder", decoder, TrainModelPartConfig, False))

        # decoder text encoder
        decoder_text_encoder = TrainModelPartConfig.default_values()
        data.append(("decoder_text_encoder", decoder_text_encoder, TrainModelPartConfig, False))

        # decoder vqgan
        decoder_vqgan = TrainModelPartConfig.default_values()
        data.append(("decoder_vqgan", decoder_vqgan, TrainModelPartConfig, False))

        # masked training
        data.append(("masked_training", False, bool, False))
        data.append(("unmasked_probability", 0.1, float, False))
        data.append(("unmasked_weight", 0.1, float, False))
        data.append(("normalize_masked_area_loss", False, bool, False))
        data.append(("masked_prior_preservation_weight", 0.0, float, False))
        data.append(("custom_conditioning_image", False, bool, False))

        #layer filter
        data.append(("layer_filter", "", str, False))
        data.append(("layer_filter_preset", "full", str, False))
        data.append(("layer_filter_regex", False, bool, False))

        # embedding
        data.append(("embedding_learning_rate", None, float, True))
        data.append(("preserve_embedding_norm", False, bool, False))
        data.append(("embedding", TrainEmbeddingConfig.default_values(), TrainEmbeddingConfig, False))
        data.append(("additional_embeddings", [], list[TrainEmbeddingConfig], False))
        data.append(("embedding_weight_dtype", DataType.FLOAT_32, DataType, False))

        # cloud
        data.append(("cloud", CloudConfig.default_values(), CloudConfig, False))

        # lora
        data.append(("peft_type", PeftType.LORA, PeftType, False))
        data.append(("lora_model_name", "", str, False))
        data.append(("lora_rank", 16, int, False))
        data.append(("lora_alpha", 1.0, float, False))
        data.append(("lora_decompose", False, bool, False))
        data.append(("lora_decompose_norm_epsilon", True, bool, False))
        data.append(("lora_decompose_output_axis", False, bool, False))
        data.append(("lora_weight_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("bundle_additional_embeddings", True, bool, False))

        # oft
        data.append(("oft_block_size", 32, int, False))
        data.append(("oft_coft", False, bool, False))
        data.append(("coft_eps", 1e-4, float, False))
        data.append(("oft_block_share", False, bool, False))

        # optimizer
        data.append(("optimizer", TrainOptimizerConfig.default_values(), TrainOptimizerConfig, False))
        data.append(("optimizer_defaults", {}, dict[str, TrainOptimizerConfig], False))

        # sample settings
        data.append(("sample_definition_file_name", "training_samples/samples.json", str, False))
        data.append(("samples", None, list[SampleConfig], True))
        data.append(("sample_after", 10, int, False))
        data.append(("sample_after_unit", TimeUnit.MINUTE, TimeUnit, False))
        data.append(("sample_skip_first", 0, int, False))
        data.append(("sample_image_format", ImageFormat.JPG, ImageFormat, False))
        data.append(("sample_video_format", VideoFormat.MP4, VideoFormat, False))
        data.append(("sample_audio_format", AudioFormat.MP3, AudioFormat, False))
        data.append(("samples_to_tensorboard", True, bool, False))
        data.append(("non_ema_sampling", True, bool, False))

        # backup settings
        data.append(("backup_after", 30, int, False))
        data.append(("backup_after_unit", TimeUnit.MINUTE, TimeUnit, False))
        data.append(("rolling_backup", False, bool, False))
        data.append(("rolling_backup_count", 3, int, False))
        data.append(("backup_before_save", True, bool, False))
        data.append(("save_every", 0, int, False))
        data.append(("save_every_unit", TimeUnit.NEVER, TimeUnit, False))
        data.append(("save_skip_first", 0, int, False))
        data.append(("save_filename_prefix", "", str, False))

        # secrets
        secrets = SecretsConfig.default_values()
        data.append(("secrets", secrets, SecretsConfig, False))

        return TrainConfig(data)
