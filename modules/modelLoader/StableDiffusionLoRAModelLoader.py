import json
import os

import torch

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.modelLoader.stableDiffusion.StableDiffusionLoRALoader import StableDiffusionLoRALoader
from modules.modelLoader.stableDiffusion.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionLoRAModelLoader(
    BaseModelLoader,
    ModelLoaderModelSpecMixin,
    InternalModelLoaderMixin,
):
    def __init__(self):
        super(StableDiffusionLoRAModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/sd_model_spec/sd_1.5-lora.json"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/sd_model_spec/sd_1.5_inpainting-lora.json"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/sd_model_spec/sd_2.0-lora.json"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/sd_model_spec/sd_2.0-lora.json"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/sd_model_spec/sd_2.0_inpainting-lora.json"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/sd_model_spec/sd_2.0_depth-lora.json"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/sd_model_spec/sd_2.1-lora.json"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/sd_model_spec/sd_2.1-lora.json"
            case _:
                return None

    def __load_internal(
            self,
            model: StableDiffusionModel,
            weight_dtypes: ModelWeightDtypes,
            lora_name: str,
    ):
        with open(os.path.join(lora_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )

        # embedding model
        pt_lora_name = os.path.join(lora_name, "lora", "lora.pt")
        safetensors_lora_name = os.path.join(lora_name, "lora", "lora.safetensors")
        if os.path.exists(pt_lora_name):
            self.__load_ckpt(model, weight_dtypes, pt_lora_name)
        elif os.path.exists(safetensors_lora_name):
            self.__load_safetensors(model, weight_dtypes, safetensors_lora_name)
        else:
            raise Exception("no lora found")

        # optimizer
        try:
            model.optimizer_state_dict = torch.load(os.path.join(lora_name, "optimizer", "optimizer.pt"))
        except FileNotFoundError:
            pass

        # ema
        try:
            model.ema_state_dict = torch.load(os.path.join(lora_name, "ema", "ema.pt"))
        except FileNotFoundError:
            pass

        # meta
        model.train_progress = train_progress

        # model spec
        model.model_spec = self._load_default_model_spec(model.model_type)

        return True

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> StableDiffusionModel | None:
        base_model_loader = StableDiffusionModelLoader()
        lora_model_loader = StableDiffusionLoRALoader()

        model = StableDiffusionModel(model_type=model_type)

        if model_names.base_model is not None:
            model = base_model_loader.load(model, model_type, model_names, weight_dtypes)
        lora_model_loader.load(model, model_names)
        self._load_internal_data(model, model_names.base_model)

        model.model_spec = self._load_default_model_spec(model_type)

        return model