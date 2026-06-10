from enum import Enum


class ModelFormat(Enum):
    DIFFUSERS = 'DIFFUSERS'
    CKPT = 'CKPT'
    SAFETENSORS = 'SAFETENSORS'
    LEGACY_SAFETENSORS = 'LEGACY_SAFETENSORS'
    COMFY_LORA = 'COMFY_LORA'

    INTERNAL = 'INTERNAL'  # an internal format that stores all information to resume training

    def __str__(self):
        return self.value


    def file_extension(self) -> str:
        match self:
            case ModelFormat.DIFFUSERS:
                return ''
            case ModelFormat.CKPT:
                return '.ckpt'
            case ModelFormat.SAFETENSORS:
                return '.safetensors'
            case ModelFormat.LEGACY_SAFETENSORS:
                return '.safetensors'
            case ModelFormat.COMFY_LORA:
                return '.safetensors'
            case _:
                return ''

    def is_single_file(self) -> bool:
        return self.file_extension() != ''
