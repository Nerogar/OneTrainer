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

    STABLE_DIFFUSION_XL_10_BASE = 'STABLE_DIFFUSION_XL_10_BASE'
    STABLE_DIFFUSION_XL_10_BASE_INPAINTING = 'STABLE_DIFFUSION_XL_10_BASE_INPAINTING'

    KANDINSKY_21 = 'KANDINSKY_21'

    WUERSTCHEN_2 = 'WUERSTCHEN_2'

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

    def is_kandinsky(self):
        return self == ModelType.KANDINSKY_21

    def is_wuerstchen(self):
        return self == ModelType.WUERSTCHEN_2

    def has_mask_input(self) -> bool:
        return self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING

    def has_conditioning_image_input(self) -> bool:
        return self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_20_INPAINTING \
            or self == ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING

    def has_depth_input(self):
        return self == ModelType.STABLE_DIFFUSION_20_DEPTH

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
