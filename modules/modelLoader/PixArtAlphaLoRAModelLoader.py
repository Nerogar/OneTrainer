import json
import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.PixArtAlphaModelLoader import PixArtAlphaModelLoader
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class PixArtAlphaLoRAModelLoader(BaseModelLoader, ModelSpecModelLoaderMixin):
    def __init__(self):
        super(PixArtAlphaLoRAModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.PIXART_ALPHA:
                return "resources/sd_model_spec/pixart_alpha_1.0-lora.json"
            case _:
                return None

    def __init_lora(
            self,
            model: PixArtAlphaModel,
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

        model.transformer_lora = self._load_lora_with_prefix(
            module=model.transformer,
            state_dict=state_dict,
            prefix="lora_transformer",
            rank=rank,
            module_filter=["attn1", "attn2"],
        )

    def __load_safetensors(
            self,
            model: PixArtAlphaModel,
            weight_dtypes: ModelWeightDtypes,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type, lora_name)

        state_dict = load_file(lora_name)
        self.__init_lora(model, state_dict, weight_dtypes.lora.torch_dtype())

    def __load_ckpt(
            self,
            model: PixArtAlphaModel,
            weight_dtypes: ModelWeightDtypes,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type)

        state_dict = torch.load(lora_name)
        self.__init_lora(model, state_dict, weight_dtypes.lora.torch_dtype())

    def __load_internal(
            self,
            model: PixArtAlphaModel,
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
    ) -> PixArtAlphaModel | None:
        stacktraces = []

        base_model_loader = PixArtAlphaModelLoader()

        if model_names.base_model is not None:
            model = base_model_loader.load(model_type, model_names, weight_dtypes)
        else:
            model = PixArtAlphaModel(model_type=model_type)

        if model_names.lora:
            try:
                self.__load_internal(model, weight_dtypes, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_ckpt(model, weight_dtypes, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_safetensors(model, weight_dtypes, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())
        else:
            return model

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
