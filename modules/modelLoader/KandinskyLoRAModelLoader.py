import json
import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.KandinskyModel import KandinskyModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.KandinskyModelLoader import KandinskyModelLoader
from modules.modelLoader.mixin.ModelLoaderLoRAMixin import ModelLoaderLoRAMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class KandinskyLoRAModelLoader(BaseModelLoader, ModelLoaderLoRAMixin):
    def __init__(self):
        super(KandinskyLoRAModelLoader, self).__init__()

    def __init_lora(
            self,
            model: KandinskyModel,
            state_dict: dict[str, Tensor],
    ):
        rank = self._get_lora_rank(state_dict)

        model.unet_lora = self._load_lora_with_prefix(
            module=model.unet,
            state_dict=state_dict,
            prefix="lora_unet",
            rank=rank,
            module_filter=["attentions"],
        )

    def __load_safetensors(
            self,
            model: KandinskyModel,
            lora_name: str,
    ):
        state_dict = load_file(lora_name)
        self.__init_lora(model, state_dict)

    def __load_ckpt(
            self,
            model: KandinskyModel,
            lora_name: str,
    ):
        state_dict = torch.load(lora_name)
        self.__init_lora(model, state_dict)

    def __load_internal(
            self,
            model: KandinskyModel,
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

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> KandinskyModel | None:
        stacktraces = []

        base_model_loader = KandinskyModelLoader()

        if model_names.base_model:
            model = base_model_loader.load(model_type, model_names, weight_dtypes)
        else:
            model = KandinskyModel(model_type=model_type)

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

        return model
