import json
import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.WuerstchenModelLoader import WuerstchenModelLoader
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.convert.convert_stable_cascade_lora_ckpt_to_diffusers import \
    convert_stable_cascade_lora_ckpt_to_diffusers
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class WuerstchenLoRAModelLoader(BaseModelLoader, ModelSpecModelLoaderMixin):
    def __init__(self):
        super(WuerstchenLoRAModelLoader, self).__init__()

    def __init_lora(
            self,
            model: WuerstchenModel,
            state_dict: dict[str, Tensor],
    ):
        rank = self._get_lora_rank(state_dict)

        model.prior_text_encoder_lora = self._load_lora_with_prefix(
            module=model.prior_text_encoder,
            state_dict=state_dict,
            prefix="lora_prior_te",
            rank=rank,
        )

        model.prior_prior_lora = self._load_lora_with_prefix(
            module=model.prior_prior,
            state_dict=state_dict,
            prefix="lora_prior_unet",
            rank=rank,
            module_filter=["attention"],
        )

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.WUERSTCHEN_2:
                return "resources/sd_model_spec/wuerstchen_2.0-lora.json"
            case ModelType.STABLE_CASCADE_1:
                return "resources/sd_model_spec/stable_cascade_1.0-lora.json"
            case _:
                return None

    def __load_safetensors(
            self,
            model: WuerstchenModel,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type, lora_name)

        state_dict = load_file(lora_name)
        if model.model_type.is_stable_cascade():
            state_dict = convert_stable_cascade_lora_ckpt_to_diffusers(state_dict)
        self.__init_lora(model, state_dict)

    def __load_ckpt(
            self,
            model: WuerstchenModel,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type)

        state_dict = torch.load(lora_name)
        if model.model_type.is_stable_cascade():
            state_dict = convert_stable_cascade_lora_ckpt_to_diffusers(state_dict)
        self.__init_lora(model, state_dict)

    def __load_internal(
            self,
            model: WuerstchenModel,
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
            self.__load_ckpt(model, pt_lora_name)
        elif os.path.exists(safetensors_lora_name):
            self.__load_safetensors(model, safetensors_lora_name)
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
        try:
            with open(os.path.join(lora_name, "model_spec.json"), "r") as model_spec_file:
                model.model_spec = ModelSpec.from_dict(json.load(model_spec_file))
        except:
            pass

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> WuerstchenModel | None:
        stacktraces = []

        base_model_loader = WuerstchenModelLoader()

        if model_names.base_model is not None:
            model = base_model_loader.load(model_type, model_names, weight_dtypes)
        else:
            model = WuerstchenModel(model_type=model_type)

        if model_names.lora:
            try:
                self.__load_internal(model, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_ckpt(model, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_safetensors(model, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())
        else:
            return model

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
