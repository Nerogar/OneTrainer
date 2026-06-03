from enum import Enum

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

    Z_IMAGE = 'Z_IMAGE'

    ERNIE = 'ERNIE'

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
        return self.is_stable_diffusion_3() \
            or self.is_stable_diffusion_xl() \
            or self.is_flux_1() \
            or self.is_hunyuan_video() \
            or self.is_hi_dream() \

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
            or self.is_sana() \
            or self.is_hunyuan_video() \
            or self.is_hi_dream() \
            or self.is_z_image() \
            or self.is_ernie()

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
        if self.is_qwen() or self.is_anima() or self.is_z_image() or self.is_flux_2() or self.is_ernie():
            return (TrainingMethod.FINE_TUNE, TrainingMethod.LORA)
        raise ValueError(f"No supported training methods defined for model type {self}")


# The first text encoder is always "text_encoder" here (matching the config field), even for
# multi-encoder models that refer to it as "text_encoder_1" elsewhere in the code.
_MODEL_PARTS: dict[ModelType, tuple[str, ...]] = {
    ModelType.STABLE_DIFFUSION_15: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_15_INPAINTING: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_20: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_20_BASE: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_20_INPAINTING: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_20_DEPTH: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_21: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_21_BASE: ("text_encoder", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_3: ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"),
    ModelType.STABLE_DIFFUSION_35: ("text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae"),
    ModelType.STABLE_DIFFUSION_XL_10_BASE: ("text_encoder", "text_encoder_2", "unet", "vae"),
    ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING: ("text_encoder", "text_encoder_2", "unet", "vae"),
    # Only Würstchen v2's decoder has its own text encoder; Stable Cascade's decoder does not.
    ModelType.WUERSTCHEN_2: ("text_encoder", "prior", "effnet_encoder", "decoder", "decoder_text_encoder", "decoder_vqgan"),
    ModelType.STABLE_CASCADE_1: ("text_encoder", "prior", "effnet_encoder", "decoder", "decoder_vqgan"),
    ModelType.PIXART_ALPHA: ("text_encoder", "transformer", "vae"),
    ModelType.PIXART_SIGMA: ("text_encoder", "transformer", "vae"),
    ModelType.FLUX_DEV_1: ("text_encoder", "text_encoder_2", "transformer", "vae"),
    ModelType.FLUX_FILL_DEV_1: ("text_encoder", "text_encoder_2", "transformer", "vae"),
    ModelType.FLUX_2: ("text_encoder", "transformer", "vae"),
    ModelType.SANA: ("text_encoder", "transformer", "vae"),
    ModelType.HUNYUAN_VIDEO: ("text_encoder", "text_encoder_2", "transformer", "vae"),
    ModelType.HI_DREAM_FULL: ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "transformer", "vae"),
    ModelType.CHROMA_1: ("text_encoder", "transformer", "vae"),
    ModelType.QWEN: ("text_encoder", "transformer", "vae"),
    ModelType.ANIMA: ("text_encoder", "transformer", "vae"),
    ModelType.Z_IMAGE: ("text_encoder", "transformer", "vae"),
    ModelType.ERNIE: ("text_encoder", "transformer", "vae"),
}


class PeftType(Enum):
    LORA = 'LORA'
    LOHA = 'LOHA'
    OFT_2 = 'OFT_2'
    LOKR = 'LOKR'

    def __str__(self):
        return self.value
