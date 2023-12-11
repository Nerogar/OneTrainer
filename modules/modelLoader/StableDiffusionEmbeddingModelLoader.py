import json
import os
import traceback

import torch
from safetensors.torch import load_file

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionEmbeddingModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin):
    def __init__(self):
        super(StableDiffusionEmbeddingModelLoader, self).__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_15:
                return "resources/sd_model_spec/sd_1.5-embedding.json"
            case ModelType.STABLE_DIFFUSION_15_INPAINTING:
                return "resources/sd_model_spec/sd_1.5_inpainting-embedding.json"
            case ModelType.STABLE_DIFFUSION_20:
                return "resources/sd_model_spec/sd_2.0-embedding.json"
            case ModelType.STABLE_DIFFUSION_20_BASE:
                return "resources/sd_model_spec/sd_2.0-embedding.json"
            case ModelType.STABLE_DIFFUSION_20_INPAINTING:
                return "resources/sd_model_spec/sd_2.0_inpainting-embedding.json"
            case ModelType.STABLE_DIFFUSION_20_DEPTH:
                return "resources/sd_model_spec/sd_2.0_depth-embedding.json"
            case ModelType.STABLE_DIFFUSION_21:
                return "resources/sd_model_spec/sd_2.1-embedding.json"
            case ModelType.STABLE_DIFFUSION_21_BASE:
                return "resources/sd_model_spec/sd_2.1-embedding.json"
            case _:
                return None

    def __load_ckpt(
            self,
            model: StableDiffusionModel,
            embedding_names: list[str],
    ):
        embedding_name = embedding_names[0]

        embedding_state = torch.load(embedding_name)

        string_to_param_dict = embedding_state['string_to_param']

        embedding = StableDiffusionModelEmbedding(
            text_encoder_vector=string_to_param_dict['*'],
            prefix='embedding',
        )

        model.embeddings = [embedding]
        model.model_spec = self._load_default_model_spec(model.model_type)

    def __load_safetensors(
            self,
            model: StableDiffusionModel,
            embedding_names: list[str],
    ):
        embedding_name = embedding_names[0]

        embedding_state = load_file(embedding_name)

        tensor = embedding_state["emp_params"]

        embedding = StableDiffusionModelEmbedding(
            text_encoder_vector=tensor,
            prefix='embedding',
        )

        model.embeddings = [embedding]
        model.model_spec = self._load_default_model_spec(model.model_type, embedding_name)

    def __load_internal(
            self,
            model: StableDiffusionModel,
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
    ) -> StableDiffusionModel | None:
        stacktraces = []

        base_model_loader = StableDiffusionModelLoader()

        if model_names.base_model is not None:
            model = base_model_loader.load(model_type, model_names, weight_dtypes)
        else:
            model = StableDiffusionModel(model_type=model_type)

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
        raise Exception("could not load embedding: " + str(model_names.embedding))
