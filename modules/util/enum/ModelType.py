from enum import Enum

from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod


class ModelType(Enum):
    STABLE_DIFFUSION_15 = 'STABLE_DIFFUSION_15'
    STABLE_DIFFUSION_15_INPAINTING = 'STABLE_DIFFUSION_15_INPAINTING'
    STABLE_DIFFUSION_20 = 'STABLE_DIFFUSION_20'
    STABLE_DIFFUSION_20_BASE = 'STABLE_DIFFUSION_20_BASE'
    STABLE_DIFFUSION_20_INPAINTING = 'STABLE_DIFFUSION_20_INPAINTING'
    STABLE_DIFFUSION_20_DEPTH = 'STABLE_DIFFUSION_20_DEPTH'
    STABLE_DIFFUSION_21 = 'STABLE_DIFFUSION_21'
    STABLE_DIFFUSION_21_BASE = 'STABLE_DIFFUSION_21_BASE'

    STABLE_DIFFUSION_3 = 'STABLE_DIFFUSION_3'
    STABLE_DIFFUSION_35 = 'STABLE_DIFFUSION_35'

    STABLE_DIFFUSION_XL_10_BASE = 'STABLE_DIFFUSION_XL_10_BASE'
    STABLE_DIFFUSION_XL_10_BASE_INPAINTING = 'STABLE_DIFFUSION_XL_10_BASE_INPAINTING'

    WUERSTCHEN_2 = 'WUERSTCHEN_2'
    STABLE_CASCADE_1 = 'STABLE_CASCADE_1'

    PIXART_ALPHA = 'PIXART_ALPHA'
    PIXART_SIGMA = 'PIXART_SIGMA'

    FLUX_DEV_1 = 'FLUX_DEV_1'
    FLUX_FILL_DEV_1 = 'FLUX_FILL_DEV_1'
    FLUX_2 = 'FLUX_2'

    SANA = 'SANA'

    HUNYUAN_VIDEO = 'HUNYUAN_VIDEO'

    HI_DREAM_FULL = 'HI_DREAM_FULL'

    CHROMA_1 = 'CHROMA_1'

    QWEN = 'QWEN'

    ANIMA = 'ANIMA'
    KREA_2 = 'KREA_2'

    Z_IMAGE = 'Z_IMAGE'

    ERNIE = 'ERNIE'

    IDEOGRAM_4 = 'IDEOGRAM_4'

    def __str__(self):
        return self.value

    def is_stable_diffusion(self):
        return self == ModelType.STABLE_DIFFUSION_15 \
            or self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20 \
            or self == ModelType.STABLE_DIFFUSION_20_BASE \
            or self == ModelType.STABLE_DIFFUSION_20_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20_DEPTH \
            or self == ModelType.STABLE_DIFFUSION_21 \
            or self == ModelType.STABLE_DIFFUSION_21_BASE

    def is_stable_diffusion_xl(self):
        return self == ModelType.STABLE_DIFFUSION_XL_10_BASE \
            or self == ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING

    def is_stable_diffusion_3(self):
        return self == ModelType.STABLE_DIFFUSION_3 \
            or self == ModelType.STABLE_DIFFUSION_35

    def is_stable_diffusion_3_5(self):
        return self == ModelType.STABLE_DIFFUSION_35

    def is_wuerstchen(self):
        return self == ModelType.WUERSTCHEN_2 \
            or self == ModelType.STABLE_CASCADE_1

    def is_pixart(self):
        return self == ModelType.PIXART_ALPHA \
            or self == ModelType.PIXART_SIGMA

    def is_pixart_alpha(self):
        return self == ModelType.PIXART_ALPHA

    def is_pixart_sigma(self):
        return self == ModelType.PIXART_SIGMA

    def is_flux(self):
        return self == ModelType.FLUX_DEV_1 \
            or self == ModelType.FLUX_FILL_DEV_1 \
            or self == ModelType.FLUX_2

    def is_flux_1(self):
        return self == ModelType.FLUX_DEV_1 \
            or self == ModelType.FLUX_FILL_DEV_1

    def is_flux_2(self):
        return self == ModelType.FLUX_2

    def is_chroma(self):
        return self == ModelType.CHROMA_1

    def is_qwen(self):
        return self == ModelType.QWEN

    def is_anima(self):
        return self == ModelType.ANIMA

    def is_krea2(self):
        return self == ModelType.KREA_2

    def is_sana(self):
        return self == ModelType.SANA

    def is_hunyuan_video(self):
        return self == ModelType.HUNYUAN_VIDEO

    def is_hi_dream(self):
        return self == ModelType.HI_DREAM_FULL

    def is_z_image(self):
        return self == ModelType.Z_IMAGE

    def is_ernie(self):
        return self == ModelType.ERNIE

    def is_ideogram(self):
        return self == ModelType.IDEOGRAM_4

    def supports_negative_prompt(self) -> bool:
        # asymmetric dual-network CFG models drive the negative branch from a frozen unconditional network (or an
        # empty prompt), not a user-supplied negative prompt
        return not self.is_ideogram()

    def has_mask_input(self) -> bool:
        return self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING \
            or self == ModelType.FLUX_FILL_DEV_1

    def has_conditioning_image_input(self) -> bool:
        return self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING \
            or self == ModelType.FLUX_FILL_DEV_1

    def has_depth_input(self):
        return self == ModelType.STABLE_DIFFUSION_20_DEPTH

    def has_multiple_text_encoders(self):
        return "text_encoder_2" in self.model_parts()

    def is_sd_v1(self):
        return self == ModelType.STABLE_DIFFUSION_15 \
            or self == ModelType.STABLE_DIFFUSION_15_INPAINTING

    def is_sd_v2(self):
        return self == ModelType.STABLE_DIFFUSION_20 \
            or self == ModelType.STABLE_DIFFUSION_20_BASE \
            or self == ModelType.STABLE_DIFFUSION_20_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20_DEPTH \
            or self == ModelType.STABLE_DIFFUSION_21 \
            or self == ModelType.STABLE_DIFFUSION_21_BASE

    def is_wuerstchen_v2(self):
        return self == ModelType.WUERSTCHEN_2

    def is_stable_cascade(self):
        return self == ModelType.STABLE_CASCADE_1

    def is_flow_matching(self) -> bool:
        return self.is_stable_diffusion_3() \
            or self.is_flux() \
            or self.is_chroma() \
            or self.is_qwen() \
            or self.is_anima() \
            or self.is_krea2() \
            or self.is_sana() \
            or self.is_hunyuan_video() \
            or self.is_hi_dream() \
            or self.is_z_image() \
            or self.is_ernie() \
            or self.is_ideogram()

    def is_video_model(self) -> bool:
        return self.is_hunyuan_video() #incase we add more video models in the future

    def model_parts(self) -> tuple[str, ...]:
        return _MODEL_PARTS[self]

    def supported_training_methods(self) -> tuple[TrainingMethod, ...]:
        if self.is_stable_diffusion():
            return (TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING, TrainingMethod.FINE_TUNE_VAE)
        if self.is_stable_diffusion_3() \
                or self.is_stable_diffusion_xl() \
                or self.is_wuerstchen() \
                or self.is_pixart() \
                or self.is_flux_1() \
                or self.is_sana() \
                or self.is_hunyuan_video() \
                or self.is_hi_dream() \
                or self.is_chroma():
            return (TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING)
        if self.is_qwen() or self.is_z_image() or self.is_flux_2() or self.is_ernie() \
                or self.is_anima() or self.is_krea2() or self.is_ideogram():
            return (TrainingMethod.FINE_TUNE, TrainingMethod.LORA)
        raise ValueError(f"No supported training methods defined for model type {self}")

    def denoising_model_part(self) -> str:
        # the denoising model component (unet / transformer / prior), always listed first in model_parts().
        return _MODEL_PARTS[self][0]

    def text_encoder_parts(self) -> tuple[str, ...]:
        # the text encoder components, named "text_encoder"/"text_encoder_2"/... by convention (see below).
        return tuple(part for part in _MODEL_PARTS[self] if part.startswith("text_encoder"))

    def trainable_parts(self) -> tuple[str, ...]:
        return _TRAINABLE_PARTS[self]

    def supported_lora_formats(self) -> list[ModelFormat]:
        formats = [
            ModelFormat.DIFFUSERS_LORA,
            ModelFormat.KOHYA_LORA,
            ModelFormat.ORIGINAL_LORA,
            ModelFormat.COMFY_LORA,
        ]
        has_legacy = self.is_stable_diffusion() \
            or self.is_stable_diffusion_xl() \
            or self.is_stable_diffusion_3() \
            or self.is_stable_cascade() \
            or self.is_pixart() \
            or self.is_flux_1() \
            or self.is_flux_2() \
            or self.is_chroma() \
            or self.is_qwen() \
            or self.is_hunyuan_video() \
            or self.is_z_image() \
            or self.is_ernie()
        if has_legacy:
            formats.append(ModelFormat.LEGACY_LORA)
        return formats

    def supported_full_model_formats(self) -> list[ModelFormat]:
        formats = [ModelFormat.DIFFUSERS]
        if self.is_stable_diffusion() or self.is_stable_diffusion_xl() or self.is_stable_diffusion_3():
            formats.append(ModelFormat.ORIGINAL_SINGLE_FILE)
        elif (self.is_flux_1() or self.is_flux_2() or self.is_chroma() or self.is_hunyuan_video()
                or self.is_hi_dream() or self.is_pixart() or self.is_qwen() or self.is_ernie()
                or self.is_z_image() or self.is_anima() or self.is_krea2() or self.is_ideogram()):
            formats.append(ModelFormat.ORIGINAL_TRANSFORMER)
        if self.is_z_image():
            formats.append(ModelFormat.COMFY_TRANSFORMER)
        has_legacy = self.is_stable_diffusion() \
            or self.is_stable_diffusion_xl() \
            or self.is_stable_diffusion_3() \
            or self.is_stable_cascade() \
            or self.is_pixart() \
            or self.is_flux_1() \
            or self.is_flux_2() \
            or self.is_chroma() \
            or self.is_qwen() \
            or self.is_hunyuan_video() \
            or self.is_hi_dream() \
            or self.is_z_image() \
            or self.is_ernie()
        if has_legacy:
            formats.append(ModelFormat.LEGACY_SAFETENSORS)
        return formats

    def supported_output_formats(self, training_method: TrainingMethod) -> list[ModelFormat]:
        if training_method == TrainingMethod.EMBEDDING:
            return [ModelFormat.SAFETENSORS]
        elif training_method == TrainingMethod.LORA:
            return self.supported_lora_formats()
        elif training_method in (TrainingMethod.FINE_TUNE, TrainingMethod.FINE_TUNE_VAE):
            return self.supported_full_model_formats()
        else:
            raise ValueError(f"Unsupported training method: {training_method}")

# The components each model type has, keyed by TrainConfig field names, as the single source of truth.
# The diffusion model (unet / transformer / prior) is always listed first; the first text encoder is
# "text_encoder" (matching the config field), even for multi-encoder models that refer to it as
# "text_encoder_1" elsewhere in the code.
_MODEL_PARTS: dict[ModelType, tuple[str, ...]] = {
    ModelType.STABLE_DIFFUSION_15: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_15_INPAINTING: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_20: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_20_BASE: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_20_INPAINTING: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_20_DEPTH: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_21: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_21_BASE: ("unet", "text_encoder", "vae"),
    ModelType.STABLE_DIFFUSION_3: ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"),
    ModelType.STABLE_DIFFUSION_35: ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"),
    ModelType.STABLE_DIFFUSION_XL_10_BASE: ("unet", "text_encoder", "text_encoder_2", "vae"),
    ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING: ("unet", "text_encoder", "text_encoder_2", "vae"),
    # Only Würstchen v2's decoder has its own text encoder; Stable Cascade's decoder does not.
    ModelType.WUERSTCHEN_2: ("prior", "text_encoder", "effnet_encoder", "decoder", "decoder_text_encoder", "decoder_vqgan"),
    ModelType.STABLE_CASCADE_1: ("prior", "text_encoder", "effnet_encoder", "decoder", "decoder_vqgan"),
    ModelType.PIXART_ALPHA: ("transformer", "text_encoder", "vae"),
    ModelType.PIXART_SIGMA: ("transformer", "text_encoder", "vae"),
    ModelType.FLUX_DEV_1: ("transformer", "text_encoder", "text_encoder_2", "vae"),
    ModelType.FLUX_FILL_DEV_1: ("transformer", "text_encoder", "text_encoder_2", "vae"),
    ModelType.FLUX_2: ("transformer", "text_encoder", "vae"),
    ModelType.ANIMA: ("transformer", "text_encoder", "vae"),
    ModelType.SANA: ("transformer", "text_encoder", "vae"),
    ModelType.HUNYUAN_VIDEO: ("transformer", "text_encoder", "text_encoder_2", "vae"),
    ModelType.HI_DREAM_FULL: ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "vae"),
    ModelType.CHROMA_1: ("transformer", "text_encoder", "vae"),
    ModelType.QWEN: ("transformer", "text_encoder", "vae"),
    ModelType.KREA_2: ("transformer", "text_encoder", "vae"),
    ModelType.Z_IMAGE: ("transformer", "text_encoder", "vae"),
    ModelType.ERNIE: ("transformer", "text_encoder", "vae"),
    ModelType.IDEOGRAM_4: ("transformer", "text_encoder", "unconditional_transformer", "vae"),
}

# subset of _MODEL_PARTS the architecture allows a run to train, for both LoRA and fine-tuning -- the parts each setup
# routes through _setup_model_part_requires_grad. Parts omitted here (VAE everywhere; the text encoder on the newer
# transformer models; Ideogram's unconditional_transformer; Wuerstchen's decoder stack) are architecture-frozen.
_TRAINABLE_PARTS: dict[ModelType, tuple[str, ...]] = {
    ModelType.STABLE_DIFFUSION_15: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_15_INPAINTING: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_20: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_20_BASE: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_20_INPAINTING: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_20_DEPTH: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_21: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_21_BASE: ("unet", "text_encoder"),
    ModelType.STABLE_DIFFUSION_3: ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3"),
    ModelType.STABLE_DIFFUSION_35: ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3"),
    ModelType.STABLE_DIFFUSION_XL_10_BASE: ("unet", "text_encoder", "text_encoder_2"),
    ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING: ("unet", "text_encoder", "text_encoder_2"),
    ModelType.WUERSTCHEN_2: ("prior", "text_encoder"),
    ModelType.STABLE_CASCADE_1: ("prior", "text_encoder"),
    ModelType.PIXART_ALPHA: ("transformer", "text_encoder"),
    ModelType.PIXART_SIGMA: ("transformer", "text_encoder"),
    ModelType.FLUX_DEV_1: ("transformer", "text_encoder", "text_encoder_2"),
    ModelType.FLUX_FILL_DEV_1: ("transformer", "text_encoder", "text_encoder_2"),
    ModelType.FLUX_2: ("transformer",),
    ModelType.ANIMA: ("transformer",),
    ModelType.SANA: ("transformer", "text_encoder"),
    ModelType.HUNYUAN_VIDEO: ("transformer", "text_encoder", "text_encoder_2"),
    ModelType.HI_DREAM_FULL: ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"),
    ModelType.CHROMA_1: ("transformer", "text_encoder"),
    ModelType.QWEN: ("transformer", "text_encoder"),
    ModelType.KREA_2: ("transformer",),
    ModelType.Z_IMAGE: ("transformer",),
    ModelType.ERNIE: ("transformer",),
    ModelType.IDEOGRAM_4: ("transformer",),
}


class PeftType(Enum):
    LORA = 'LORA'
    LOHA = 'LOHA'
    OFT_2 = 'OFT_2'
    LOKR = 'LOKR'

    def __str__(self):
        return self.value
