from enum import Enum


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

    SANA = 'SANA'

    HUNYUAN_VIDEO = 'HUNYUAN_VIDEO'

    HI_DREAM_FULL = 'HI_DREAM_FULL'

    CHROMA_1 = 'CHROMA_1'

    QWEN = 'QWEN'

    Z_IMAGE = 'Z_IMAGE'

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
            or self == ModelType.FLUX_FILL_DEV_1

    def is_chroma(self):
        return self == ModelType.CHROMA_1

    def is_qwen(self):
        return self == ModelType.QWEN

    def is_sana(self):
        return self == ModelType.SANA

    def is_hunyuan_video(self):
        return self == ModelType.HUNYUAN_VIDEO

    def is_hi_dream(self):
        return self == ModelType.HI_DREAM_FULL

    def is_z_image(self):
        return self == ModelType.Z_IMAGE

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
            or self.is_flux() \
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
            or self.is_sana() \
            or self.is_hunyuan_video() \
            or self.is_hi_dream()


class PeftType(Enum):
    LORA = 'LORA'
    LOHA = 'LOHA'
    OFT_2 = 'OFT_2'
    LOKR = 'LOKR'

    def __str__(self):
        return self.value
