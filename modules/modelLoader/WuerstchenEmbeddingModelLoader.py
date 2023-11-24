import json
import os
import traceback

import torch
from safetensors.torch import load_file

from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenModelEmbedding
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.WuerstchenModelLoader import WuerstchenModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class WuerstchenEmbeddingModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin):
    def __init__(self):
        super(WuerstchenEmbeddingModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.WUERSTCHEN_2:
                return "resources/sd_model_spec/wuerstchen_2.0-embedding.json"
            case _:
                return None

    def __load_ckpt(
            self,
            model: WuerstchenModel,
            embedding_names: list[str],
    ):
        embedding_name = embedding_names[0]

        embedding_state = torch.load(embedding_name)

        tensor = embedding_state["prior"]

        embedding = WuerstchenModelEmbedding(
            name="*",
            prior_text_encoder_vector=tensor,
            token_count=tensor.shape[0]
        )

        model.embeddings = [embedding]
        model.model_spec = self._load_default_model_spec(model.model_type)

    def __load_safetensors(
            self,
            model: WuerstchenModel,
            embedding_names: list[str],
    ):
        embedding_name = embedding_names[0]

        embedding_state = load_file(embedding_name)

        tensor = embedding_state["prior"]

        embedding = WuerstchenModelEmbedding(
            name="*",
            prior_text_encoder_vector=tensor,
            token_count=tensor.shape[0]
        )

        model.embeddings = [embedding]
        model.model_spec = self._load_default_model_spec(model.model_type, embedding_name)

    def __load_internal(
            self,
            model: WuerstchenModel,
            embedding_names: list[str],
    ):
        embedding_name = embedding_names[0]

        with open(os.path.join(embedding_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )

        # embedding model
        pt_embedding_name = os.path.join(embedding_name, "embedding", "embedding.pt")
        safetensors_embedding_name = os.path.join(embedding_name, "embedding", "embedding.safetensors")
        if os.path.exists(pt_embedding_name):
            self.__load_ckpt(model, [pt_embedding_name])
        elif os.path.exists(safetensors_embedding_name):
            self.__load_safetensors(model, [safetensors_embedding_name])
        else:
            raise Exception("no embedding found")

        # optimizer
        try:
            model.optimizer_state_dict = torch.load(os.path.join(embedding_name, "optimizer", "optimizer.pt"))
        except FileNotFoundError:
            pass

        # ema
        try:
            model.ema_state_dict = torch.load(os.path.join(embedding_name, "ema", "ema.pt"))
        except FileNotFoundError:
            pass

        # meta
        model.train_progress = train_progress
        model.model_spec = self._load_default_model_spec(model.model_type)

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

        if model_names.embedding:
            try:
                self.__load_internal(model, model_names.embedding)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_safetensors(model, model_names.embedding)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_ckpt(model, model_names.embedding)
                return model
            except:
                stacktraces.append(traceback.format_exc())
        else:
            return model

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + str(model_names.embedding))
