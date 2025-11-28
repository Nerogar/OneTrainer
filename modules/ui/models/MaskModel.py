from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.ui.models.SingletonConfigModel import SingletonConfigModel
from modules.util.enum.GenerateMasksModel import GenerateMasksAction, GenerateMasksModel
from modules.util.torch_util import default_device, torch_gc

import torch


class MaskModel(SingletonConfigModel):
    def __init__(self):
        super().__init__({
            "model": GenerateMasksModel.CLIPSEG,
            "path": "",
            "prompt": "",
            "mode": GenerateMasksAction.REPLACE,
            "alpha": 1.0,
            "threshold": 0.3,
            "smooth": 5,
            "expand": 10,
            "include_subdirectories": False,
        })

        self.masking_model = None

    def create_masks(self, progress_fn=None):
        with self.critical_region_read():
            self.__load_masking_model(self.get_state("model"))

            self.masking_model.mask_folder(
                sample_dir=self.get_state("path"),
                prompts=[self.get_state("prompt")],
                mode=str(self.get_state("mode")).lower(),
                alpha=float(self.get_state("alpha")),
                threshold=float(self.get_state("threshold")),
                smooth_pixels=int(self.get_state("smooth")),
                expand_pixels=int(self.get_state("expand")),
                include_subdirectories=self.get_state("include_subdirectories"),
                progress_callback=self.__wrap_progress(progress_fn),
            )

    def __wrap_progress(self, fn):
        def f(value, max_value):
            if fn is not None:
                fn({"value": value, "max_value": max_value})
        return f

    def __load_masking_model(self, model):
        if model == GenerateMasksModel.CLIPSEG:
            if self.masking_model is None or not isinstance(self.masking_model, ClipSegModel):
                self.log("info", "Loading ClipSeg model, this may take a while")
                self.release_model()
                self.masking_model = ClipSegModel(default_device, torch.float32)
        elif model == GenerateMasksModel.REMBG:
            if self.masking_model is None or not isinstance(self.masking_model, RembgModel):
                self.log("info", "Loading Rembg model, this may take a while")
                self.release_model()
                self.masking_model = RembgModel(default_device, torch.float32)
        elif model == GenerateMasksModel.REMBG_HUMAN:
            if self.masking_model is None or not isinstance(self.masking_model, RembgHumanModel):
                self.log("info", "Loading Rembg-Human model, this may take a while")
                self.release_model()
                self.masking_model = RembgHumanModel(default_device, torch.float32)
        elif model == GenerateMasksModel.COLOR:
            if self.masking_model is None or not isinstance(self.masking_model, MaskByColor):
                self.release_model()
                self.masking_model = MaskByColor(default_device, torch.float32)

    def release_model(self):
        """Release all models from VRAM"""
        freed = False
        if self.masking_model is not None:
            self.masking_model = None
            freed = True
        if freed:
            torch_gc()
