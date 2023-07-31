import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class StableDiffusionXLLoRAModelLoader(BaseModelLoader):
    def __init__(self):
        super(StableDiffusionXLLoRAModelLoader, self).__init__()

    @staticmethod
    def __get_rank(state_dict: dict) -> int:
        for name, state in state_dict.items():
            if "lora_down.weight" in name:
                return state.shape[0]

    @staticmethod
    def __init_lora(model: StableDiffusionXLModel, state_dict: dict[str, Tensor]):
        rank = StableDiffusionXLLoRAModelLoader.__get_rank(state_dict)

        # TODO: only create the lora if state_dict contains the keys
        model.text_encoder_1_lora = LoRAModuleWrapper(
            orig_module=model.text_encoder_1,
            rank=rank,
            prefix="lora_te1",
        ).to(dtype=torch.float32)
        model.text_encoder_1_lora.load_state_dict(state_dict)

        model.text_encoder_2_lora = LoRAModuleWrapper(
            orig_module=model.text_encoder_2,
            rank=rank,
            prefix="lora_te2",
        ).to(dtype=torch.float32)
        model.text_encoder_2_lora.load_state_dict(state_dict)

        model.unet_lora = LoRAModuleWrapper(
            orig_module=model.unet,
            rank=rank,
            prefix="lora_unet",
            module_filter=["attentions"],
        ).to(dtype=torch.float32)
        model.unet_lora.load_state_dict(state_dict)

    @staticmethod
    def __default_model_spec_name(model_type: ModelType) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/sd_model_spec/sd_xl_base_1.0_lora.json"
            case _:
                return None

    @staticmethod
    def _create_default_model_spec(
            model_type: ModelType,
    ) -> ModelSpec:
        with open(StableDiffusionXLLoRAModelLoader.__default_model_spec_name(model_type), "r") as model_spec_file:
            return ModelSpec.from_dict(json.load(model_spec_file))

    @staticmethod
    def __load_safetensors(model: StableDiffusionXLModel, lora_name: str) -> bool:
        try:
            model.model_spec = StableDiffusionXLLoRAModelLoader._create_default_model_spec(model.model_type)

            with safe_open(lora_name, framework="pt") as f:
                if "modelspec.sai_model_spec" in f.metadata():
                    model.model_spec = ModelSpec.from_dict(f.metadata())

            state_dict = load_file(lora_name)
            StableDiffusionXLLoRAModelLoader.__init_lora(model, state_dict)
            return True
        except:
            return False

    @staticmethod
    def __load_ckpt(model: StableDiffusionXLModel, lora_name: str) -> bool:
        try:
            model.model_spec = StableDiffusionXLLoRAModelLoader._create_default_model_spec(model.model_type)

            state_dict = torch.load(lora_name)
            StableDiffusionXLLoRAModelLoader.__init_lora(model, state_dict)
            return True
        except:
            return False

    @staticmethod
    def __load_internal(model: StableDiffusionXLModel, lora_name: str) -> bool:
        try:
            with open(os.path.join(lora_name, "meta.json"), "r") as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )

            # embedding model
            loaded = StableDiffusionXLLoRAModelLoader.__load_ckpt(
                model,
                os.path.join(lora_name, "lora", "lora.pt")
            )
            if not loaded:
                return False

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
            model.model_spec = StableDiffusionXLLoRAModelLoader._create_default_model_spec(model.model_type)
            try:
                with open(os.path.join(lora_name, "model_spec.json"), "r") as model_spec_file:
                    model.model_spec = ModelSpec.from_dict(json.load(model_spec_file))
            except:
                pass

            return True
        except:
            return False

    def load(
            self,
            model_type: ModelType,
            weight_dtype: torch.dtype,
            base_model_name: str,
            extra_model_name: str | None
    ) -> StableDiffusionXLModel | None:
        base_model_loader = StableDiffusionXLModelLoader()
        model = base_model_loader.load(model_type, weight_dtype, base_model_name, None)

        lora_loaded = self.__load_internal(model, extra_model_name)

        if not lora_loaded:
            lora_loaded = self.__load_ckpt(model, extra_model_name)

        if not lora_loaded:
            lora_loaded = self.__load_safetensors(model, extra_model_name)

        return model
