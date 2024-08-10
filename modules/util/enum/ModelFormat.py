from enum import Enum


class ModelFormat(Enum):
    DIFFUSERS = 'DIFFUSERS'
    CKPT = 'CKPT'
    SAFETENSORS = 'SAFETENSORS'

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
            case _:
                return ''