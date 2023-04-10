from enum import Enum


class ModelType(Enum):
    STABLE_DIFFUSION_15 = 'STABLE_DIFFUSION_15'
    STABLE_DIFFUSION_15_INPAINTING = 'STABLE_DIFFUSION_15_INPAINTING'
    STABLE_DIFFUSION_20 = 'STABLE_DIFFUSION_20'
    STABLE_DIFFUSION_20_INPAINTING = 'STABLE_DIFFUSION_20_INPAINTING'
    STABLE_DIFFUSION_20_DEPTH = 'STABLE_DIFFUSION_20_DEPTH'
    STABLE_DIFFUSION_21 = 'STABLE_DIFFUSION_21'

    def __str__(self):
        return self.value

    def is_stable_diffusion(self):
        return self == ModelType.STABLE_DIFFUSION_15 \
               or ModelType.STABLE_DIFFUSION_15_INPAINTING \
               or ModelType.STABLE_DIFFUSION_20 \
               or ModelType.STABLE_DIFFUSION_20_INPAINTING \
               or ModelType.STABLE_DIFFUSION_20_DEPTH \
               or ModelType.STABLE_DIFFUSION_21

    def has_mask_input(self) -> bool:
        return self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
               or self == ModelType.STABLE_DIFFUSION_20_INPAINTING

    def has_conditioning_image_input(self) -> bool:
        return self == ModelType.STABLE_DIFFUSION_15_INPAINTING \
               or self == ModelType.STABLE_DIFFUSION_20_INPAINTING

    def has_depth_input(self):
        return self == ModelType.STABLE_DIFFUSION_20_DEPTH
