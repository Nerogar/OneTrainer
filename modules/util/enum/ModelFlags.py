from enum import Flag, auto

from modules.modelSetup.BaseChromaSetup import PRESETS as chroma_presets
from modules.modelSetup.BaseFluxSetup import PRESETS as flux_presets
from modules.modelSetup.BaseHiDreamSetup import PRESETS as hidream_presets
from modules.modelSetup.BaseHunyuanVideoSetup import PRESETS as hunyuan_video_presets
from modules.modelSetup.BasePixArtAlphaSetup import PRESETS as pixart_presets
from modules.modelSetup.BaseQwenSetup import PRESETS as qwen_presets
from modules.modelSetup.BaseSanaSetup import PRESETS as sana_presets
from modules.modelSetup.BaseStableDiffusion3Setup import PRESETS as sd3_presets
from modules.modelSetup.BaseStableDiffusionSetup import PRESETS as sd_presets
from modules.modelSetup.BaseStableDiffusionXLSetup import PRESETS as sdxl_presets
from modules.modelSetup.BaseWuerstchenSetup import PRESETS as sc_presets
from modules.util.enum.TrainingMethod import TrainingMethod


class ModelFlags(Flag):
    NONE = 0  # Invalid initial value.

    # Model + training flags.
    UNET = auto()
    PRIOR = auto()
    OVERRIDE_PRIOR = auto()
    TRANSFORMER = auto()
    OVERRIDE_TRANSFORMER = auto()
    OVERRIDE_TE4 = auto()
    TE1 = auto()
    TE2 = auto()
    TE3 = auto()
    TE4 = auto()
    VAE = auto()

    # Model-only flags.
    EFFNET = auto()
    DEC = auto()
    DEC_TE = auto()
    ALLOW_SAFETENSORS = auto()
    ALLOW_DIFFUSERS = auto()
    ALLOW_LEGACY_SAFETENSORS = auto()

    # Training-only flags.
    TRAIN_TRANSFORMER = auto()
    TRAIN_PRIOR = auto()
    GENERALIZED_OFFSET_NOISE = auto()
    TE_INCLUDE = auto()
    VB_LOSS = auto()
    GUIDANCE_SCALE = auto()
    DYNAMIC_TIMESTEP_SHIFTING = auto()
    DISABLE_FORCE_ATTN_MASK = auto()
    DISABLE_CLIP_SKIP = auto()
    VIDEO_TRAINING = auto()
    DISABLE_TE4_LAYER_SKIP = auto()
    OVERRIDE_SEQUENCE_LENGTH_TE2 = auto()

    # Training method flags.
    CAN_TRAIN_EMBEDDING = auto()
    CAN_FINE_TUNE_VAE = auto()

    @staticmethod
    def getFlags(model_type, training_method):
        flags = ModelFlags.NONE

        if model_type.is_stable_diffusion():  # TODO simplify
            flags = ModelFlags.UNET | ModelFlags.TE1 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS | ModelFlags.GENERALIZED_OFFSET_NOISE | ModelFlags.EFFNET | ModelFlags.CAN_FINE_TUNE_VAE | ModelFlags.CAN_TRAIN_EMBEDDING
            if training_method in [TrainingMethod.FINE_TUNE, TrainingMethod.FINE_TUNE_VAE]:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_stable_diffusion_3():
            flags = ModelFlags.TE1 | ModelFlags.TE2 | ModelFlags.TE3 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS | ModelFlags.TE_INCLUDE | ModelFlags.TRANSFORMER
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_stable_diffusion_xl():
            flags = ModelFlags.UNET | ModelFlags.TE1 | ModelFlags.TE2 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS | ModelFlags.GENERALIZED_OFFSET_NOISE
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_wuerstchen():
            flags = ModelFlags.PRIOR | ModelFlags.TE1 | ModelFlags.TRAIN_TRANSFORMER | ModelFlags.DEC
            if model_type.is_stable_cascade():
                flags |= ModelFlags.OVERRIDE_PRIOR
            else:
                flags |= ModelFlags.DEC_TE
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method != TrainingMethod.FINE_TUNE or model_type.is_stable_cascade():
                flags |= ModelFlags.ALLOW_SAFETENSORS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_pixart():
            flags = ModelFlags.TRANSFORMER | ModelFlags.TE1 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS | ModelFlags.TRAIN_TRANSFORMER | ModelFlags.VB_LOSS
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_flux():
            flags = (ModelFlags.OVERRIDE_TRANSFORMER | ModelFlags.TE1 | ModelFlags.TE2 | ModelFlags.VAE | ModelFlags.OVERRIDE_SEQUENCE_LENGTH_TE2 |
                     ModelFlags.ALLOW_SAFETENSORS | ModelFlags.TRANSFORMER | ModelFlags.TE_INCLUDE | ModelFlags.GUIDANCE_SCALE | ModelFlags.DYNAMIC_TIMESTEP_SHIFTING)
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_chroma():
            flags = (ModelFlags.OVERRIDE_TRANSFORMER | ModelFlags.TE1 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS |
                     ModelFlags.DISABLE_FORCE_ATTN_MASK | ModelFlags.TRANSFORMER)
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_qwen():
            flags = (ModelFlags.OVERRIDE_TRANSFORMER | ModelFlags.TE1 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS |
                     ModelFlags.DISABLE_FORCE_ATTN_MASK | ModelFlags.TRANSFORMER | ModelFlags.DYNAMIC_TIMESTEP_SHIFTING | ModelFlags.DISABLE_CLIP_SKIP)
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_sana():
            flags = ModelFlags.TRANSFORMER | ModelFlags.TE1 | ModelFlags.VAE | ModelFlags.TRAIN_TRANSFORMER
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            else:
                flags |= ModelFlags.ALLOW_SAFETENSORS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_hunyuan_video():
            flags = (ModelFlags.TE1 | ModelFlags.TE2 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS |
                     ModelFlags.TE_INCLUDE | ModelFlags.VIDEO_TRAINING | ModelFlags.TRANSFORMER | ModelFlags.GUIDANCE_SCALE)
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        elif model_type.is_hi_dream():
            flags = (ModelFlags.OVERRIDE_TE4 | ModelFlags.TE1 | ModelFlags.TE2 | ModelFlags.TE3 | ModelFlags.TE4 | ModelFlags.VAE | ModelFlags.ALLOW_SAFETENSORS |
                     ModelFlags.TRANSFORMER | ModelFlags.VIDEO_TRAINING | ModelFlags.DISABLE_TE4_LAYER_SKIP | ModelFlags.TE_INCLUDE)
            if training_method == TrainingMethod.FINE_TUNE:
                flags |= ModelFlags.ALLOW_DIFFUSERS
            if training_method == TrainingMethod.LORA:
                flags |= ModelFlags.ALLOW_LEGACY_SAFETENSORS

        if model_type.is_stable_diffusion_3() \
        or model_type.is_stable_diffusion_xl() \
        or model_type.is_wuerstchen() \
        or model_type.is_pixart() \
        or model_type.is_flux() \
        or model_type.is_sana() \
        or model_type.is_hunyuan_video() \
        or model_type.is_hi_dream() \
        or model_type.is_chroma():
            flags |= ModelFlags.CAN_TRAIN_EMBEDDING

        return flags

    @staticmethod
    def getPresets(model_type):
        if model_type.is_stable_diffusion(): #TODO simplify
            presets = sd_presets
        elif model_type.is_stable_diffusion_xl():
            presets = sdxl_presets
        elif model_type.is_stable_diffusion_3():
            presets = sd3_presets
        elif model_type.is_wuerstchen():
            presets = sc_presets
        elif model_type.is_pixart():
            presets = pixart_presets
        elif model_type.is_flux():
            presets = flux_presets
        elif model_type.is_qwen():
            presets = qwen_presets
        elif model_type.is_chroma():
            presets = chroma_presets
        elif model_type.is_sana():
            presets = sana_presets
        elif model_type.is_hunyuan_video():
            presets = hunyuan_video_presets
        elif model_type.is_hi_dream():
            presets = hidream_presets
        else:
            presets = {"full": []}

        return presets
