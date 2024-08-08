from modules.module.BaseRembgModel import BaseRembgModel

import torch


class RembgModel(BaseRembgModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            model_filename="u2net.onnx",
            model_path="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
            model_md5="md5:60024c5c889badc19c04ad937298a77b",
            device=device,
            dtype=dtype,
        )
