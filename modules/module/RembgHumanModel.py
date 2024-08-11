from modules.module.BaseRembgModel import BaseRembgModel

import torch


class RembgHumanModel(BaseRembgModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            model_filename="u2net_human_seg.onnx",
            model_path="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
            model_md5="md5:c09ddc2e0104f800e3e1bb4652583d1f",
            device=device,
            dtype=dtype,
        )
