from enum import Enum


class ModelFormat(Enum):
    # full model
    DIFFUSERS = 'DIFFUSERS'  # diffusers pipeline, all components, multi-folder directory
    ORIGINAL_SINGLE_FILE = 'ORIGINAL_SINGLE_FILE'  # original/native namespace, single file, all components
    ORIGINAL_TRANSFORMER = 'ORIGINAL_TRANSFORMER'  # original/native namespace, single file, transformer only
    COMFY_TRANSFORMER = 'COMFY_TRANSFORMER'  # ORIGINAL_TRANSFORMER + per-model Comfy key quirks (Z-Image: qkv-fuse + norm swap)
    LEGACY_SAFETENSORS = 'LEGACY_SAFETENSORS'  # full model: per-model historical safetensors output

    # LoRA -- every value is single-save-type and _LORA-suffixed
    DIFFUSERS_LORA = 'DIFFUSERS_LORA'  # diffusers-PEFT keys (lora_A/lora_B, folded alpha)
    KOHYA_LORA = 'KOHYA_LORA'  # kohya-flat namespace, lora_down/lora_up, kept .alpha
    ORIGINAL_LORA = 'ORIGINAL_LORA'  # original/native namespace, fused qkv
    COMFY_LORA = 'COMFY_LORA'  # original namespace + diffusion_model. transformer prefix
    LEGACY_LORA = 'LEGACY_LORA'  # per-model historical LoRA output (mostly the mixture format; Cascade native / identity differ)

    # embedding
    SAFETENSORS = 'SAFETENSORS'  # embedding vectors file

    INTERNAL = 'INTERNAL'  # an internal format that stores all information to resume training

    def __str__(self):
        return self.value


    def file_extension(self) -> str:
        # context-free: each value maps to exactly one container. DIFFUSERS / INTERNAL are directories;
        # everything else is a single .safetensors file. (LEGACY_SAFETENSORS is .safetensors for every
        # model except the multi-file Stable Cascade -- a known, accepted imperfection.)
        if self in (ModelFormat.DIFFUSERS, ModelFormat.INTERNAL):
            return ''
        return '.safetensors'

    def is_single_file(self) -> bool:
        return self.file_extension() != ''

    def needs_qkv_fusion(self) -> bool:
        # these LoRA formats export native fused qkv, so the wrapper must build a fused adapter over
        # the split q/k/v (ORIGINAL/COMFY: native namespace; KOHYA: native kohya-ss layout).
        # DIFFUSERS/LEGACY keep qkv split.
        return self in (ModelFormat.ORIGINAL_LORA, ModelFormat.COMFY_LORA, ModelFormat.KOHYA_LORA)
