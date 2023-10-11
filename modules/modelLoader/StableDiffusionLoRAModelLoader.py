import json
import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.modelLoader.mixin.ModelLoaderLoRAMixin import ModelLoaderLoRAMixin
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionLoRAModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin, ModelLoaderLoRAMixin):
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

    def __init_lora(
            self,
            model: StableDiffusionModel,
            state_dict: dict[str, Tensor],
            dtype: torch.dtype,
    ):
        rank = self._get_lora_rank(state_dict)

        model.text_encoder_lora = self._load_lora_with_prefix(
            module=model.text_encoder,
            state_dict=state_dict,
            prefix="lora_te",
            rank=rank,
        )

        model.unet_lora = self._load_lora_with_prefix(
            module=model.unet,
            state_dict=state_dict,
            prefix="lora_unet",
            rank=rank,
            module_filter=["attentions"],
        )

    def __load_safetensors(
            self,
            model: StableDiffusionModel,
            weight_dtypes: ModelWeightDtypes,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type, lora_name)

        state_dict = load_file(lora_name)
        self.__init_lora(model, state_dict, weight_dtypes.lora.torch_dtype())

    def __load_ckpt(
            self,
            model: StableDiffusionModel,
            weight_dtypes: ModelWeightDtypes,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type)

        state_dict = torch.load(lora_name)
        self.__init_lora(model, state_dict, weight_dtypes.lora.torch_dtype())

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
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str | None,
            extra_model_name: str | None
    ) -> StableDiffusionModel | None:
        stacktraces = []

        base_model_loader = StableDiffusionModelLoader()

        if base_model_name is not None:
            model = base_model_loader.load(model_type, weight_dtypes, base_model_name, None)
        else:
            model = StableDiffusionModel(model_type=model_type)

        if extra_model_name:
            try:
                self.__load_internal(model, weight_dtypes, extra_model_name)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_ckpt(model, weight_dtypes, extra_model_name)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_safetensors(model, weight_dtypes, extra_model_name)
                return model
            except:
                stacktraces.append(traceback.format_exc())
        else:
            return model

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + extra_model_name)
