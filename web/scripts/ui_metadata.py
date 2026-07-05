ENUM_DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "ModelType": {
        "STABLE_DIFFUSION_15": "SD1.5",
        "STABLE_DIFFUSION_15_INPAINTING": "SD1.5 Inpainting",
        "STABLE_DIFFUSION_20": "SD2.0",
        "STABLE_DIFFUSION_20_BASE": "SD2.0 Base",
        "STABLE_DIFFUSION_20_INPAINTING": "SD2.0 Inpainting",
        "STABLE_DIFFUSION_20_DEPTH": "SD2.0 Depth",
        "STABLE_DIFFUSION_21": "SD2.1",
        "STABLE_DIFFUSION_21_BASE": "SD2.1 Base",
        "STABLE_DIFFUSION_3": "SD3",
        "STABLE_DIFFUSION_35": "SD3.5",
        "STABLE_DIFFUSION_XL_10_BASE": "SDXL",
        "STABLE_DIFFUSION_XL_10_BASE_INPAINTING": "SDXL Inpainting",
        "WUERSTCHEN_2": "Wuerstchen v2",
        "STABLE_CASCADE_1": "Stable Cascade",
        "PIXART_ALPHA": "PixArt Alpha",
        "PIXART_SIGMA": "PixArt Sigma",
        "FLUX_DEV_1": "Flux Dev.1",
        "FLUX_FILL_DEV_1": "Flux Fill Dev",
        "FLUX_2": "Flux 2 [Dev, Klein]",
        "SANA": "Sana",
        "HUNYUAN_VIDEO": "Hunyuan Video",
        "HI_DREAM_FULL": "HiDream Full",
        "CHROMA_1": "Chroma1",
        "QWEN": "QwenImage",
        "Z_IMAGE": "Z-Image",
    },
    "TrainingMethod": {
        "FINE_TUNE": "Fine-Tune",
        "LORA": "LoRA",
        "EMBEDDING": "Embedding",
        "FINE_TUNE_VAE": "Fine-Tune VAE",
    },
    "DataType": {
        "NONE": "None",
        "FLOAT_8": "Float 8",
        "FLOAT_16": "Float 16",
        "FLOAT_32": "Float 32",
        "BFLOAT_16": "BFloat 16",
        "TFLOAT_32": "TFloat 32",
        "INT_8": "Int 8",
        "NFLOAT_4": "NFloat 4",
        "FLOAT_W8A8": "Float W8A8",
        "INT_W8A8": "Int W8A8",
        "GGUF": "GGUF",
        "GGUF_A8_FLOAT": "GGUF A8 Float",
        "GGUF_A8_INT": "GGUF A8 Int",
    },
    "PeftType": {
        "LORA": "LoRA",
        "LOHA": "LoHa",
        "OFT_2": "OFT v2",
    },
    "EMAMode": {
        "OFF": "Off",
        "GPU": "GPU",
        "CPU": "CPU",
    },
    "GradientCheckpointingMethod": {
        "OFF": "Off",
        "ON": "On",
        "CPU_OFFLOADED": "CPU Offloaded",
    },
    "GradientReducePrecision": {
        "WEIGHT_DTYPE": "Weight Dtype",
        "WEIGHT_DTYPE_STOCHASTIC": "Weight Dtype Stochastic",
        "FLOAT_32": "Float 32",
        "FLOAT_32_STOCHASTIC": "Float 32 Stochastic",
    },
    "BalancingStrategy": {
        "REPEATS": "Repeats",
        "SAMPLES": "Samples",
    },
    "ConceptType": {
        "STANDARD": "Standard",
        "VALIDATION": "Validation",
        "PRIOR_PREDICTION": "Prior Prediction",
    },
    "ConfigPart": {
        "NONE": "None",
        "SETTINGS": "Settings",
        "ALL": "All",
    },
    "CloudAction": {
        "NONE": "None",
        "STOP": "Stop",
        "DELETE": "Delete",
    },
    "CloudFileSync": {
        "FABRIC_SFTP": "Fabric SFTP",
        "NATIVE_SCP": "Native SCP",
    },
    "CloudType": {
        "RUNPOD": "RunPod",
        "LINUX": "Linux",
    },
    "FileType": {
        "IMAGE": "Image",
        "VIDEO": "Video",
        "AUDIO": "Audio",
    },
    "GenerateCaptionsModel": {
        "BLIP": "BLIP",
        "BLIP2": "BLIP2",
        "WD14_VIT_2": "WD14 ViT 2",
    },
    "GenerateMasksModel": {
        "CLIPSEG": "CLIPSeg",
        "REMBG": "RemBG",
        "REMBG_HUMAN": "RemBG Human",
        "COLOR": "Color",
    },
    "ImageFormat": {
        "PNG": "PNG",
        "JPG": "JPG",
    },
    "AudioFormat": {
        "MP3": "MP3",
        "MP4": "MP4",
    },
    "VideoFormat": {
        "PNG_IMAGE_SEQUENCE": "PNG Image Sequence",
        "JPG_IMAGE_SEQUENCE": "JPG Image Sequence",
    },
    "LearningRateScaler": {
        "NONE": "None",
        "BATCH": "Batch",
        "GLOBAL_BATCH": "Global Batch",
        "GRADIENT_ACCUMULATION": "Gradient Accumulation",
        "BOTH": "Both",
        "GLOBAL_BOTH": "Global Both",
    },
    "LossScaler": {
        "NONE": "None",
        "BATCH": "Batch",
        "GLOBAL_BATCH": "Global Batch",
        "GRADIENT_ACCUMULATION": "Gradient Accumulation",
        "BOTH": "Both",
        "GLOBAL_BOTH": "Global Both",
    },
    "LearningRateScheduler": {
        "CONSTANT": "Constant",
        "LINEAR": "Linear",
        "COSINE": "Cosine",
        "COSINE_WITH_RESTARTS": "Cosine with Restarts",
        "COSINE_WITH_HARD_RESTARTS": "Cosine with Hard Restarts",
        "REX": "Rex",
        "ADAFACTOR": "Adafactor",
        "CUSTOM": "Custom",
    },
    "LossWeight": {
        "NONE": "None",
        "P2": "P2",
        "MIN_SNR_GAMMA": "Min SNR Gamma",
        "DEBIASED_ESTIMATION": "Debiased Estimation",
        "SIGMA": "Sigma",
    },
    "ModelFormat": {
        "DIFFUSERS": "Diffusers",
        "CKPT": "Checkpoint",
        "SAFETENSORS": "Safetensors",
        "LEGACY_SAFETENSORS": "Legacy Safetensors",
        "COMFY_LORA": "Comfy LoRA",
        "INTERNAL": "Internal",
    },
    "NoiseScheduler": {
        "DDIM": "DDIM",
        "EULER": "Euler",
        "EULER_A": "Euler A",
        "DPMPP": "DPM++",
        "DPMPP_SDE": "DPM++ SDE",
        "UNIPC": "UniPC",
        "EULER_KARRAS": "Euler Karras",
        "DPMPP_KARRAS": "DPM++ Karras",
        "DPMPP_SDE_KARRAS": "DPM++ SDE Karras",
        "UNIPC_KARRAS": "UniPC Karras",
    },
    "Optimizer": {
        "ADAGRAD": "Adagrad",
        "ADAGRAD_8BIT": "Adagrad 8-bit",
        "ADAM": "Adam",
        "ADAM_8BIT": "Adam 8-bit",
        "ADAMW": "AdamW",
        "ADAMW_8BIT": "AdamW 8-bit",
        "ADAMW_ADV": "AdamW Adv",
        "AdEMAMix": "AdEMAMix",
        "AdEMAMix_8BIT": "AdEMAMix 8-bit",
        "SIMPLIFIED_AdEMAMix": "Simplified AdEMAMix",
        "ADOPT": "ADOPT",
        "ADOPT_ADV": "ADOPT Adv",
        "LAMB": "LAMB",
        "LAMB_8BIT": "LAMB 8-bit",
        "LARS": "LARS",
        "LARS_8BIT": "LARS 8-bit",
        "LION": "LION",
        "LION_8BIT": "LION 8-bit",
        "LION_ADV": "LION Adv",
        "RMSPROP": "RMSProp",
        "RMSPROP_8BIT": "RMSProp 8-bit",
        "SGD": "SGD",
        "SGD_8BIT": "SGD 8-bit",
        "SIGNSGD_ADV": "SignSGD Adv",
        "SCHEDULE_FREE_ADAMW": "Schedule-Free AdamW",
        "SCHEDULE_FREE_SGD": "Schedule-Free SGD",
        "DADAPT_ADA_GRAD": "D-Adapt AdaGrad",
        "DADAPT_ADAM": "D-Adapt Adam",
        "DADAPT_ADAN": "D-Adapt Adan",
        "DADAPT_LION": "D-Adapt LION",
        "DADAPT_SGD": "D-Adapt SGD",
        "PRODIGY": "Prodigy",
        "PRODIGY_PLUS_SCHEDULE_FREE": "Prodigy+ Schedule-Free",
        "PRODIGY_ADV": "Prodigy Adv",
        "LION_PRODIGY_ADV": "LION Prodigy Adv",
        "ADAFACTOR": "Adafactor",
        "CAME": "CAME",
        "CAME_8BIT": "CAME 8-bit",
        "MUON": "Muon",
        "MUON_ADV": "Muon Adv",
        "ADAMUON_ADV": "AdaMuon Adv",
        "ADABELIEF": "AdaBelief",
        "TIGER": "TIGER",
        "AIDA": "AIDA",
        "YOGI": "Yogi",
    },
    "TimeUnit": {
        "EPOCH": "Epoch",
        "STEP": "Step",
        "SECOND": "Second",
        "MINUTE": "Minute",
        "HOUR": "Hour",
        "NEVER": "Never",
        "ALWAYS": "Always",
    },
    "TimestepDistribution": {
        "UNIFORM": "Uniform",
        "SIGMOID": "Sigmoid",
        "LOGIT_NORMAL": "Logit Normal",
        "HEAVY_TAIL": "Heavy Tail",
        "COS_MAP": "Cos Map",
        "INVERTED_PARABOLA": "Inverted Parabola",
    },
}

# Tool-modal dropdown options that don't map 1:1 to a Python enum.
# The caption/mask tool modals show backend-supported models/modes; the web
# frontend extends the CTk list (OpenAI/Gemini/YOLO), and backend mode keys are
# declared as unordered sets (`VALID_CAPTION_MODES`, `VALID_MASK_MODES`) in
# `web/backend/services/tool_service.py`, so a curated ordering lives here.
# Entries are (display_label, backend_value) pairs; order determines UI order.
TOOL_DROPDOWN_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "CAPTION_MODELS": [
        ("Blip", "Blip"),
        ("Blip2", "Blip2"),
        ("WD14 VIT v2", "WD14 VIT v2"),
        ("OpenAI Compatible", "OpenAI Compatible"),
        ("Gemini API", "Gemini API"),
    ],
    "CAPTION_MODES": [
        ("Replace all captions", "replace"),
        ("Create if absent", "fill"),
        ("Add as new line", "add"),
    ],
    "MASK_MODELS": [
        ("ClipSeg", "ClipSeg"),
        ("Rembg", "Rembg"),
        ("Rembg-Human", "Rembg-Human"),
        ("Hex Color", "Hex Color"),
        ("YOLO", "YOLO"),
    ],
    "MASK_MODES": [
        ("Replace all masks", "replace"),
        ("Create if absent", "fill"),
        ("Add to existing", "add"),
        ("Subtract from existing", "subtract"),
        ("Blend with existing", "blend"),
    ],
}

DTYPE_SUBSETS: dict[str, list[tuple[str, str]]] = {
    "base": [
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
        ("float16", "FLOAT_16"),
        ("float8 (W8)", "FLOAT_8"),
        ("nfloat4", "NFLOAT_4"),
    ],
    "with_a8": [
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
        ("float16", "FLOAT_16"),
        ("float8 (W8)", "FLOAT_8"),
        ("nfloat4", "NFLOAT_4"),
        ("float W8A8", "FLOAT_W8A8"),
        ("int W8A8", "INT_W8A8"),
    ],
    "with_gguf_a8": [
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
        ("float16", "FLOAT_16"),
        ("float8 (W8)", "FLOAT_8"),
        ("nfloat4", "NFLOAT_4"),
        ("float W8A8", "FLOAT_W8A8"),
        ("int W8A8", "INT_W8A8"),
        ("GGUF", "GGUF"),
        ("GGUF A8 float", "GGUF_A8_FLOAT"),
        ("GGUF A8 int", "GGUF_A8_INT"),
    ],
    "output": [
        ("float16", "FLOAT_16"),
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
        ("float8", "FLOAT_8"),
        ("nfloat4", "NFLOAT_4"),
    ],
    "train": [
        ("float32", "FLOAT_32"),
        ("float16", "FLOAT_16"),
        ("bfloat16", "BFLOAT_16"),
        ("tfloat32", "TFLOAT_32"),
    ],
    "fallback_train": [
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
    ],
    "svd_quant": [
        ("disabled", "NONE"),
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
    ],
    "lora_weight": [
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
    ],
    "embedding_weight": [
        ("float32", "FLOAT_32"),
        ("bfloat16", "BFLOAT_16"),
    ],
}

FIELD_TOOLTIPS: dict[str, str] = {
    "secrets.huggingface_token": (
        "Enter your Hugging Face access token if you have used a protected"
        " Hugging Face repository below.\nThis value is stored separately,"
        " not saved to your configuration file."
        " Go to https://huggingface.co/settings/tokens to create an access token."
    ),
    "base_model_name": "Filename, directory or Hugging Face repository of the base model",
    "compile": (
        "Uses torch.compile and Triton to significantly speed up training."
        " Only applies to transformer/unet. Disable in case of compatibility issues."
    ),

    "unet.weight_dtype": "The unet weight data type",
    "transformer.model_name": (
        "Can be used to override the transformer in the base model."
        " Safetensors and GGUF files are supported, local and on Huggingface."
        " If a GGUF file is used, the DataType must also be set to GGUF"
    ),
    "transformer.weight_dtype": "The transformer weight data type",
    "prior.model_name": "Filename, directory or Hugging Face repository of the prior model",
    "prior.weight_dtype": "The prior weight data type",

    "text_encoder.weight_dtype": "The text encoder weight data type",
    "text_encoder_2.weight_dtype": "The text encoder 2 weight data type",
    "text_encoder_3.weight_dtype": "The text encoder 3 weight data type",
    "text_encoder_4.model_name": "Filename, directory or Hugging Face repository of the text encoder 4 model",
    "text_encoder_4.weight_dtype": "The text encoder 4 weight data type",

    "vae.model_name": (
        "Directory or Hugging Face repository of a VAE model in diffusers format."
        " Can be used to override the VAE included in the base model."
        " Using a safetensor VAE file will cause an error that the model cannot be loaded."
    ),
    "vae.weight_dtype": "The vae weight data type",

    "effnet_encoder.model_name": "Filename, directory or Hugging Face repository of the effnet encoder model",
    "effnet_encoder.weight_dtype": "The effnet encoder weight data type",
    "decoder.model_name": "Filename, directory or Hugging Face repository of the decoder model",
    "decoder.weight_dtype": "The decoder weight data type",
    "decoder_text_encoder.weight_dtype": "The decoder text encoder weight data type",
    "decoder_vqgan.weight_dtype": "The decoder vqgan weight data type",

    "quantization.layer_filter_preset": (
        "Select a preset defining which layers to quantize."
        " Quantization of certain layers can decrease model quality."
        " Only applies to the transformer/unet"
    ),
    "quantization.layer_filter": (
        "Comma-separated list of layers to quantize."
        " Regular expressions (if toggled) are supported."
        " Any model layer with a matching name will be quantized"
    ),
    "quantization.layer_filter_regex": (
        "If enabled, layer filter patterns are interpreted as regular expressions."
        " Otherwise, simple substring matching is used."
    ),
    "quantization.svd_dtype": "What datatype to use for SVDQuant weights decomposition.",
    "quantization.svd_rank": "Rank for SVDQuant weights decomposition",

    "output_model_destination": "Filename or directory where the output model is saved",
    "output_dtype": "Precision to use when saving the output model",
    "output_model_format": "Format to use when saving the output model",
    "include_train_config": (
        "Include the training configuration in the final model."
        " Only supported for safetensors files."
        " None: No config is included."
        " Settings: All training settings are included."
        " All: All settings, including the samples and concepts are included."
    ),

    "optimizer.optimizer": "The type of optimizer",
    "learning_rate_scheduler": "Learning rate scheduler that automatically changes the learning rate during training",
    "learning_rate": "The base learning rate",
    "learning_rate_warmup_steps": (
        "The number of steps it takes to gradually increase the learning rate from 0"
        " to the specified learning rate. Values >1 are interpreted as a fixed number"
        " of steps, values <=1 are interpreted as a percentage of the total training"
        " steps (ex. 0.2 = 20% of the total step count)"
    ),
    "learning_rate_min_factor": (
        "Unit = float. Method = percentage. For a factor of 0.1, the final LR will be"
        " 10% of the initial LR. If the initial LR is 1e-4, the final LR will be 1e-5."
    ),
    "learning_rate_cycles": (
        "The number of learning rate cycles."
        " This is only applicable if the learning rate scheduler supports cycles"
    ),
    "epochs": "The number of epochs for a full training run",
    "batch_size": (
        "The batch size of one training step."
        " If you use multiple GPUs, this is the batch size of each GPU (local batch size)."
    ),
    "gradient_accumulation_steps": (
        "Number of accumulation steps."
        " Increase this number to trade batch size for training speed"
    ),
    "learning_rate_scaler": (
        "Selects the type of learning rate scaling to use during training."
        " Functionally equated as: LR * SQRT(selection)"
    ),
    "clip_grad_norm": "Clips the gradient norm. Leave empty to disable gradient clipping.",

    "ema": (
        "EMA averages the training progress over many steps,"
        " better preserving different concepts in big datasets"
    ),
    "ema_decay": (
        "Decay parameter of the EMA model. Higher numbers will average more steps."
        " For datasets of hundreds or thousands of images, set this to 0.9999."
        " For smaller datasets, set it to 0.999 or even 0.998"
    ),
    "ema_update_step_interval": "Number of steps between EMA update steps",
    "gradient_checkpointing": (
        "Enables gradient checkpointing."
        " This reduces memory usage, but increases training time"
    ),
    "layer_offload_fraction": (
        "Enables offloading of individual layers during training to reduce VRAM usage."
        " Increases training time and uses more RAM."
        " Only available if checkpointing is set to CPU_OFFLOADED."
        " values between 0 and 1, 0=disabled"
    ),
    "enable_async_offloading": (
        "Enable asynchronous layer offloading to overlap GPU compute with CPU-GPU transfers."
        " Can improve throughput when using layer offloading"
    ),
    "enable_activation_offloading": (
        "Offload activations to CPU during the forward pass to reduce GPU memory usage."
        " May slow training slightly due to CPU-GPU transfers"
    ),
    "train_dtype": (
        "The mixed precision data type used for training."
        " This can increase training speed, but reduces precision"
    ),
    "fallback_train_dtype": (
        "The mixed precision data type used for training stages that don't support"
        " float16 data types. This can increase training speed, but reduces precision"
    ),
    "enable_autocast_cache": (
        "Enables the autocast cache."
        " Disabling this reduces memory usage, but increases training time"
    ),
    "resolution": (
        "The resolution used for training."
        " Optionally specify multiple resolutions separated by a comma,"
        " or a single exact resolution in the format <width>x<height>"
    ),
    "frames": "The number of frames used for training.",
    "force_circular_padding": (
        "Enables circular padding for all conv layers"
        " to better train seamless images"
    ),

    "text_encoder.train": "Enables training the text encoder model",
    "text_encoder.dropout_probability": "The Probability for dropping the text encoder conditioning",
    "text_encoder.stop_training_after": "When to stop training the text encoder",
    "text_encoder.learning_rate": "The learning rate of the text encoder. Overrides the base learning rate",
    "text_encoder_layer_skip": "The number of additional clip layers to skip. 0 = the model default",
    "text_encoder_sequence_length": "Number of tokens for captions",

    "text_encoder.include": "Includes the text encoder in the training run",
    "text_encoder.train_embedding": "Enables training embeddings for the text encoder model",
    "text_encoder_2.include": "Includes text encoder 2 in the training run",
    "text_encoder_2.train": "Enables training the text encoder 2 model",
    "text_encoder_2.train_embedding": "Enables training embeddings for the text encoder 2 model",
    "text_encoder_2.dropout_probability": "The Probability for dropping the text encoder 2 conditioning",
    "text_encoder_2.stop_training_after": "When to stop training the text encoder 2",
    "text_encoder_2.learning_rate": "The learning rate of the text encoder 2. Overrides the base learning rate",
    "text_encoder_2_layer_skip": "The number of additional clip layers to skip. 0 = the model default",
    "text_encoder_2_sequence_length": (
        "Overrides the number of tokens used for captions."
        " If empty, the model default is used, which is 512 on Flux."
        " Comfy samples with 256 tokens though."
        " 77 is the default only for backwards compatibility."
    ),
    "text_encoder_3.include": "Includes text encoder 3 in the training run",
    "text_encoder_3.train": "Enables training the text encoder 3 model",
    "text_encoder_3.train_embedding": "Enables training embeddings for the text encoder 3 model",
    "text_encoder_3.dropout_probability": "The Probability for dropping the text encoder 3 conditioning",
    "text_encoder_3.stop_training_after": "When to stop training the text encoder 3",
    "text_encoder_3.learning_rate": "The learning rate of the text encoder 3. Overrides the base learning rate",
    "text_encoder_3_layer_skip": "The number of additional clip layers to skip. 0 = the model default",
    "text_encoder_4.include": "Includes text encoder 4 in the training run",
    "text_encoder_4.train": "Enables training the text encoder 4 model",
    "text_encoder_4.train_embedding": "Enables training embeddings for the text encoder 4 model",
    "text_encoder_4.dropout_probability": "The Probability for dropping the text encoder 4 conditioning",
    "text_encoder_4.stop_training_after": "When to stop training the text encoder 4",
    "text_encoder_4.learning_rate": "The learning rate of the text encoder 4. Overrides the base learning rate",

    "embedding_learning_rate": "The learning rate of embeddings. Overrides the base learning rate",
    "preserve_embedding_norm": "Rescales each trained embedding to the median embedding norm",

    "unet.train": "Enables training the UNet model",
    "unet.stop_training_after": "When to stop training the UNet",
    "unet.learning_rate": "The learning rate of the UNet. Overrides the base learning rate",
    "rescale_noise_scheduler_to_zero_terminal_snr": (
        "Rescales the noise scheduler to a zero terminal signal to noise ratio"
        " and switches the model to a v-prediction target"
    ),

    "prior.train": "Enables training the Prior model",
    "prior.stop_training_after": "When to stop training the Prior",
    "prior.learning_rate": "The learning rate of the Prior. Overrides the base learning rate",

    "transformer.train": "Enables training the Transformer model",
    "transformer.stop_training_after": "When to stop training the Transformer",
    "transformer.learning_rate": "The learning rate of the Transformer. Overrides the base learning rate",
    "transformer.attention_mask": (
        "Force enables passing of a text embedding attention mask to the transformer."
        " This can improve training on shorter captions."
    ),
    "transformer.guidance_scale": (
        "The guidance scale of guidance distilled models"
        " passed to the transformer during training."
    ),

    "offset_noise_weight": "The weight of offset noise added to each training step",
    "generalized_offset_noise": (
        "Per-timestep 'brightness knob' instead of a fixed offset -"
        " steadier training, better starts, and improved very dark/bright images."
        " Compatible with V-pred and Eps-pred. Start with 0.02 and adjust as needed."
    ),
    "perturbation_noise_weight": "The weight of perturbation noise added to each training step",
    "timestep_distribution": "Selects the function to sample timesteps during training",
    "min_noising_strength": (
        "Specifies the minimum noising strength used during training."
        " This can help to improve composition, but prevents finer details from being trained"
    ),
    "max_noising_strength": (
        "Specifies the maximum noising strength used during training."
        " This can be useful to reduce overfitting, but also reduces the impact"
        " of training samples on the overall image composition"
    ),
    "noising_weight": (
        "Controls the weight parameter of the timestep distribution function."
        " Use the preview to see more details."
    ),
    "noising_bias": (
        "Controls the bias parameter of the timestep distribution function."
        " Use the preview to see more details."
    ),
    "timestep_shift": "Shift the timestep distribution. Use the preview to see more details.",
    "dynamic_timestep_shifting": (
        "Dynamically shift the timestep distribution based on resolution."
        " If enabled, the shifting parameters are taken from the model's scheduler"
        " configuration and Timestep Shift is ignored."
        " Note: For Z-Image and Flux2, the dynamic shifting parameters are likely"
        " wrong and unknown. Use with care or set your own, fixed shift."
    ),

    "masked_training": (
        "Masks the training samples to let the model focus on certain parts"
        " of the image. When enabled, one mask image is loaded for each training sample."
    ),
    "unmasked_probability": (
        "When masked training is enabled, specifies the number of"
        " training steps done on unmasked samples"
    ),
    "unmasked_weight": (
        "When masked training is enabled, specifies the loss weight"
        " of areas outside the masked region"
    ),
    "normalize_masked_area_loss": (
        "When masked training is enabled, normalizes the loss for each sample"
        " based on the sizes of the masked region"
    ),
    "masked_prior_preservation_weight": (
        "Preserves regions outside the mask using the original untrained model"
        " output as a target. Only available for LoRA training."
        " If enabled, use a low unmasked weight."
    ),
    "custom_conditioning_image": (
        "When custom conditioning image is enabled, will use png postfix with"
        " -condlabel instead of automatically generated."
        " It's suitable for special scenarios, such as object removal,"
        " allowing the model to learn a certain behavior concept"
    ),

    "mse_strength": "Mean Squared Error strength for custom loss settings. Strengths should generally sum to 1.",
    "mae_strength": "Mean Absolute Error strength for custom loss settings. Strengths should generally sum to 1.",
    "log_cosh_strength": (
        "Log - Hyperbolic cosine Error strength for custom loss settings."
        " Strengths should generally sum to 1."
    ),
    "huber_strength": (
        "Huber loss strength for custom loss settings."
        " Less sensitive to outliers than MSE. Strengths should generally sum to 1."
    ),
    "huber_delta": "Delta parameter for huber loss",
    "vb_loss_strength": (
        "Variational lower-bound strength for custom loss settings."
        " Should be set to 1 for variational diffusion models"
    ),
    "loss_weight_fn": "Choice of loss weight function. Can help the model learn details more accurately.",
    "loss_weight_strength": (
        "Inverse strength of loss weighting."
        " Range: 1-20, only applies to Min SNR and P2."
    ),
    "loss_scaler": (
        "Selects the type of loss scaling to use during training."
        " Functionally equated as: Loss * selection"
    ),

    "layer_filter_preset": (
        "Select a preset defining which layers to train, or select 'Custom'"
        " to define your own.\nA blank 'custom' field or 'Full' will train all layers."
    ),
    "layer_filter": (
        "Comma-separated list of diffusion layers to train."
        " Regular expressions (if toggled) are supported."
        " Any model layer with a matching name will be trained"
    ),
    "layer_filter_regex": (
        "If enabled, layer filter patterns are interpreted as regular expressions."
        " Otherwise, simple substring matching is used."
    ),

    "peft_type": "The type of low-parameter finetuning method.",
    "lora_model_name": "The base LoRA to train on. Leave empty to create a new LoRA",
    "lora_rank": "The rank parameter used when creating a new LoRA",
    "lora_alpha": "The alpha parameter used when creating a new LoRA",
    "dropout_probability": (
        "Dropout probability. This percentage of model nodes will be randomly"
        " ignored at each training step. Helps with overfitting. 0 disables, 1 maximum."
    ),
    "lora_weight_dtype": (
        "The LoRA weight data type used for training."
        " This can reduce memory consumption, but reduces precision"
    ),
    "bundle_additional_embeddings": (
        "Bundles any additional embeddings into the LoRA output file,"
        " rather than as separate files"
    ),
    "lora_decompose": "Decompose LoRA Weights (aka, DoRA).",
    "lora_decompose_norm_epsilon": (
        "Add an epsilon to the norm division calculation in DoRA."
        " Can aid in training stability, and also acts as regularization."
    ),
    "lora_decompose_output_axis": (
        "Apply the weight decomposition on the output axis instead of the input axis."
    ),
    "oft_block_size": "The block size parameter used when creating a new OFT v2",
    "oft_coft": (
        "Use the constrained variant of OFT. This constrains the learned rotation"
        " to stay very close to the identity matrix, limiting adaptation to only"
        " small changes. This improves training stability, helps prevent overfitting"
        " on small datasets, and better preserves the base model's original knowledge"
        " but it may lack expressiveness for tasks requiring substantial adaptation."
    ),
    "coft_eps": "The control strength of COFT. Only has an effect if COFT is enabled.",
    "oft_block_share": (
        "Share the OFT parameters between blocks. A single rotation matrix is"
        " shared across all blocks within a layer, drastically cutting the number"
        " of trainable parameters and yielding very compact adapter files."
    ),

    "embedding.model_name": "The base embedding to train on. Leave empty to create a new embedding",
    "embedding.placeholder": "The placeholder used when using the embedding in a prompt",
    "embedding.token_count": (
        "The token count used when creating a new embedding."
        " Leave empty to auto detect from the initial embedding text."
    ),
    "embedding.initial_embedding_text": "The initial embedding text used when creating a new embedding",
    "embedding_weight_dtype": (
        "The Embedding weight data type used for training."
        " This can reduce memory consumption, but reduces precision"
    ),
    "embedding.is_output_embedding": (
        "Output embeddings are calculated at the output of the text encoder,"
        " not the input. This can improve results for larger text encoders"
        " and lower VRAM usage."
    ),
    "embedding.train": "Enable or disable training of the embedding",
    "embedding.stop_training_after": "When to stop training the embedding",

    "additional_embedding.model_name": "The base embedding to train on. Leave empty to create a new embedding",
    "additional_embedding.placeholder": "The placeholder used when using the embedding in a prompt",
    "additional_embedding.token_count": (
        "The token count used when creating a new embedding."
        " Leave empty to auto detect from the initial embedding text."
    ),
    "additional_embedding.is_output_embedding": (
        "Output embeddings are calculated at the output of the text encoder,"
        " not the input. This can improve results for larger text encoders"
        " and lower VRAM usage."
    ),
    "additional_embedding.stop_training_after": "When to stop training the embedding",
    "additional_embedding.initial_embedding_text": "The initial embedding text used when creating a new embedding",

    "workspace_dir": "The directory where all files of this training run are saved",
    "cache_dir": "The directory where cached data is saved",
    "continue_last_backup": "Automatically continues training from the last backup saved in <workspace>/backup",
    "only_cache": "Only populate the cache, without any training",

    "debug_mode": "Save debug information during the training into the debug directory",
    "debug_dir": "The directory where debug data is saved",

    "tensorboard": "Starts the Tensorboard Web UI during training",
    "tensorboard_always_on": (
        "Keep Tensorboard accessible even when not training."
        " Useful for monitoring completed training sessions."
    ),
    "tensorboard_expose": "Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)",
    "tensorboard_port": "Port to use for Tensorboard link",

    "validation": "Enable validation steps and add new graph in tensorboard",
    "validate_after": "The interval used when validate training",

    "dataloader_threads": (
        "Number of threads used for the data loader."
        " Increase if your GPU has room during caching,"
        " decrease if it's going out of memory during caching."
    ),
    "train_device": (
        'The device used for training. Can be "cuda", "cuda:0", "cuda:1" etc.'
        ' Default:"cuda". Must be "cuda" for multi-GPU training.'
    ),
    "temp_device": 'The device used to temporarily offload models while they are not used. Default:"cpu"',

    "multi_gpu": "Enable multi-GPU training",
    "device_indexes": (
        "Multi-GPU: A comma-separated list of device indexes."
        " If empty, all your GPUs are used."
        ' With a list such as "0,1,3,4" you can omit a GPU,'
        " for example an on-board graphics GPU."
    ),
    "gradient_reduce_precision": (
        "WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type;"
        " can be imprecise, but more efficient than float32\n"
        "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type,"
        " but average them in float32 and stochastically round"
        " if your weight data type is bfloat16\n"
        "FLOAT_32: Reduce gradients in float32\n"
        "FLOAT_32_STOCHASTIC: Reduce gradients in float32;"
        " use stochastic rounding to bfloat16 if your weight data type is bfloat16"
    ),
    "fused_gradient_reduce": (
        "Multi-GPU: Gradient synchronisation during the backward pass."
        " Can be more efficient, especially with Async Gradient Reduce"
    ),
    "async_gradient_reduce": (
        "Multi-GPU: Asynchronously start the gradient reduce operations"
        " during the backward pass. Can be more efficient, but requires some VRAM."
    ),
    "async_gradient_reduce_buffer": (
        'Multi-GPU: Maximum VRAM for "Async Gradient Reduce", in megabytes.'
        ' A multiple of this value can be needed if combined with "Fused Back Pass"'
        ' and/or "Layer offload fraction"'
    ),

    "sample_after": "The interval used when automatically sampling from the model during training",
    "sample_skip_first": "Start sampling automatically after this interval has elapsed.",
    "sample_image_format": "File Format used when saving samples",
    "non_ema_sampling": "Whether to include non-ema sampling when using ema.",
    "samples_to_tensorboard": "Whether to include sample images in the Tensorboard output.",

    "backup_after": "The interval used when automatically creating model backups during training",
    "rolling_backup": "If rolling backups are enabled, older backups are deleted automatically",
    "rolling_backup_count": "Defines the number of backups to keep if rolling backups are enabled",
    "backup_before_save": "Create a full backup before saving the final model",
    "save_every": "The interval used when automatically saving the model during training",
    "save_skip_first": "Start saving automatically after this interval has elapsed",
    "save_filename_prefix": "The prefix for filenames used when saving the model during training",

    "aspect_ratio_bucketing": "Aspect ratio bucketing enables training on images with different aspect ratios",
    "latent_caching": "Caching of intermediate training data that can be re-used between epochs",
    "clear_cache_before_training": (
        "Clears the cache directory before starting to train."
        " Only disable this if you want to continue using the same cached data."
        " Disabling this can lead to errors, if other settings are changed during a restart"
    ),

    "sample.frames": "Number of frames to generate. Only used when generating videos.",
    "sample.length": "Length in seconds of audio output.",
    "sample.sample_inpainting": "Enables inpainting sampling. Only available when sampling from an inpainting model.",
    "sample.base_image_path": "The base image used when inpainting.",
    "sample.mask_image_path": "The mask used when inpainting.",

    "cloud.enabled": "Enable cloud training",
    "cloud.type": (
        "Choose LINUX to connect to a linux machine via SSH."
        " Choose RUNPOD for additional functionality such as automatically creating and deleting pods."
    ),
    "cloud.file_sync": (
        "Choose NATIVE_SCP to use scp.exe to transfer files."
        " FABRIC_SFTP uses the Paramiko/Fabric SFTP implementation for file transfers instead."
    ),
    "secrets.cloud.api_key": (
        "Cloud service API key for RUNPOD. Leave empty for LINUX."
        " This value is stored separately, not saved to your configuration file."
    ),
    "secrets.cloud.host": (
        "SSH server hostname or IP."
        " Leave empty if you have a Cloud ID or want to automatically create a new cloud."
    ),
    "secrets.cloud.port": (
        "SSH server port."
        " Leave empty if you have a Cloud ID or want to automatically create a new cloud."
    ),
    "secrets.cloud.user": (
        'SSH username. Use "root" for RUNPOD.'
        " Your SSH client must be set up to connect to the cloud using a public key, without a password."
        " For RUNPOD, create an ed25519 key locally, and copy the contents of the"
        ' public keyfile to your "SSH Public Keys" on the RunPod website.'
    ),
    "secrets.cloud.key_file": (
        "Absolute path to the private key file used for SSH connections."
        " Leave empty to rely on your system SSH configuration."
    ),
    "secrets.cloud.password": (
        "SSH password for password-based authentication."
        " If you try to use native SCP requires sshpass to be installed."
        " Leave empty to use key-based authentication."
    ),
    "secrets.cloud.id": (
        "RUNPOD Cloud ID. The cloud service must have a public IP and SSH service."
        " Leave empty if you want to automatically create a new RUNPOD cloud,"
        " or if you're connecting to another cloud provider via SSH Hostname and Port."
    ),
    "cloud.tensorboard_tunnel": "Instead of starting tensorboard locally, make a TCP tunnel to a tensorboard on the cloud",
    "cloud.remote_dir": "The directory on the cloud where files will be uploaded and downloaded.",
    "cloud.onetrainer_dir": "The directory for OneTrainer on the cloud.",
    "cloud.huggingface_cache_dir": "Huggingface models are downloaded to this remote directory.",
    "cloud.install_onetrainer": "Automatically install OneTrainer from GitHub if the directory doesn't already exist.",
    "cloud.install_cmd": (
        "The command for installing OneTrainer."
        " Leave the default, unless you want to use a development branch of OneTrainer."
    ),
    "cloud.update_onetrainer": "Update OneTrainer if it already exists on the cloud.",
    "cloud.detach_trainer": "Allows the trainer to keep running even if your connection to the cloud is lost.",
    "cloud.run_id": (
        "An id identifying the remotely running trainer."
        " In case you have lost connection or closed OneTrainer,"
        " it will try to reattach to this id instead of starting a new remote trainer."
    ),
    "cloud.download_samples": "Download samples from the remote workspace directory to your local machine.",
    "cloud.download_output_model": (
        "Download the final model after training."
        " You can disable this if you plan to use an automatically saved checkpoint instead."
    ),
    "cloud.download_saves": (
        "Download the automatically saved training checkpoints from the"
        " remote workspace directory to your local machine."
    ),
    "cloud.download_backups": (
        "Download backups from the remote workspace directory to your local machine."
        " It's usually not necessary to download them, because as long as the backups"
        " are still available on the cloud, the training can be restarted using one of the cloud's backups."
    ),
    "cloud.download_tensorboard": (
        "Download TensorBoard event logs from the remote workspace directory to your local machine."
        " They can then be viewed locally in TensorBoard."
        ' It is recommended to disable "Sample to TensorBoard" to reduce the event log size.'
    ),
    "cloud.delete_workspace": (
        "Delete the workspace directory on the cloud after training has finished"
        " successfully and data has been downloaded."
    ),
    "cloud.create": (
        "Automatically creates a new cloud instance if both Host:Port and Cloud ID are empty."
        " Currently supported for RUNPOD."
    ),
    "cloud.name": "The name of the new cloud instance.",
    "cloud.sub_type": "Select the RunPod cloud type. See RunPod's website for details.",
    "cloud.gpu_type": "Select the GPU type. Enter an API key before pressing the button.",
    "cloud.volume_size": (
        "Set the storage volume size in GB."
        " This volume persists only until the cloud is deleted - not a RunPod network volume"
    ),
    "cloud.min_download": "Set the minimum download speed of the cloud in Mbps.",
    "cloud.on_finish": (
        "What to do when training finishes and the data has been fully downloaded:"
        " Stop or delete the cloud, or do nothing."
    ),
    "cloud.on_error": (
        "What to do if training stops due to an error:"
        " Stop or delete the cloud, or do nothing. Data may be lost."
    ),
    "cloud.on_detached_finish": (
        "What to do when training finishes, but the client has been detached"
        " and cannot download data. Data may be lost."
    ),
    "cloud.on_detached_error": (
        "What to do if training stops due to an error, but the client has been detached"
        " and cannot download data. Data may be lost."
    ),

    "concept.name": "Name of the concept",
    "concept.enabled": "Enable or disable this concept",
    "concept.type": (
        "STANDARD: Standard finetuning with the sample as training target\n"
        "VALIDATION: Use concept for validation instead of training\n"
        "PRIOR_PREDICTION: Use the sample to make a prediction using the model as it was"
        " before training. This prediction is then used as the training target for the"
        " model in training. This can be used as regularisation and to preserve prior model"
        " knowledge while finetuning the model on other concepts. Only implemented for LoRA."
    ),
    "concept.path": "Path where the training data is located",
    "concept.text.prompt_source": (
        "The source for prompts used during training."
        " When selecting 'From single text file', select a text file that contains a list of prompts"
    ),
    "concept.include_subdirectories": "Includes images from subdirectories into the dataset",
    "concept.image_variations": "The number of different image versions to cache if latent caching is enabled.",
    "concept.text_variations": "The number of different text versions to cache if latent caching is enabled.",
    "concept.balancing": (
        "The number of samples used during training."
        " Use repeats to multiply the concept, or samples to specify an exact number of samples used in each epoch."
    ),
    "concept.loss_weight": "The loss multiplier for this concept.",

    "concept.image.enable_crop_jitter": "Enables random cropping of samples",
    "concept.image.enable_random_flip": "Randomly flip the sample during training",
    "concept.image.enable_fixed_flip": "Apply a fixed horizontal flip to all images in this concept",
    "concept.image.enable_random_rotate": "Randomly rotates the sample during training",
    "concept.image.enable_fixed_rotate": "Apply a fixed rotation to all images in this concept",
    "concept.image.enable_random_brightness": "Randomly adjusts the brightness of the sample during training",
    "concept.image.enable_random_contrast": "Randomly adjusts the contrast of the sample during training",
    "concept.image.enable_random_saturation": "Randomly adjusts the saturation of the sample during training",
    "concept.image.enable_random_hue": "Randomly adjusts the hue of the sample during training",
    "concept.image.enable_random_circular_mask_shrink": "Automatically create circular masks for masked training",
    "concept.image.enable_random_mask_rotate_crop": "Randomly rotate the training samples and crop to the masked region",
    "concept.image.enable_resolution_override": (
        "Override the resolution for this concept."
        " Optionally specify multiple resolutions separated by a comma,"
        " or a single exact resolution in the format <width>x<height>"
    ),

    "concept.text.enable_tag_shuffling": "Enables tag shuffling",
    "concept.text.tag_delimiter": "The delimiter between tags",
    "concept.text.keep_tags_count": "The number of tags at the start of the caption that are not shuffled or dropped",
    "concept.text.tag_dropout_enable": "Enables random dropout for tags in the captions.",
    "concept.text.tag_dropout_mode": (
        "Method used to drop captions."
        " 'Full' will drop the entire caption past the 'kept' tags with a certain probability,"
        " 'Random' will drop individual tags with the set probability,"
        " and 'Random Weighted' will linearly increase the probability of dropping tags,"
        " more likely to preserve tags near the front with full probability to drop at the end."
    ),
    "concept.text.tag_dropout_probability": "Probability to drop tags, from 0 to 1.",
    "concept.text.tag_dropout_special_tags": (
        "List of tags which will be whitelisted/blacklisted by dropout."
        " 'Whitelist' tags will never be dropped but all others may be,"
        " 'Blacklist' tags may be dropped but all others will never be,"
        " 'None' may drop any tags."
        " Can specify either a delimiter-separated list in the field,"
        " or a file path to a .txt or .csv file with entries separated by newlines."
    ),
    "concept.text.tag_dropout_special_tags_mode": "Select whether special tags act as a whitelist, blacklist, or are disabled.",
    "concept.text.tag_dropout_special_tags_regex": (
        "Interpret special tags with regex, such as 'photo.*' to match"
        " 'photo, photograph, photon' but not 'telephoto'."
        " Includes exception for '/(' and '/)' syntax found in many booru/e6 tags."
    ),
    "concept.text.caps_randomize_enable": "Enables randomization of capitalization for tags in the caption.",
    "concept.text.caps_randomize_lowercase": "If enabled, converts the caption to lowercase before any further processing.",
    "concept.text.caps_randomize_mode": (
        "Comma-separated list of types of capitalization randomization to perform."
        " 'capslock' for ALL CAPS, 'title' for First Letter Of Every Word,"
        " 'first' for First word only, 'random' for randomized letters."
    ),
    "concept.text.caps_randomize_probability": "Probability to randomize capitalization of each tag, from 0 to 1.",

    "tools.dataset": "Open the captioning tool",
    "tools.video": "Open the video tools",
    "tools.convert_model": "Open the model conversion tool",
    "tools.sampling": "Open the model sampling tool",
    "tools.profiling": "Open the profiling tools.",
}

WIDE_TOOLTIPS: set[str] = {
    "secrets.huggingface_token",
    "timestep_distribution",
    "dynamic_timestep_shifting",
    "gradient_reduce_precision",
}
