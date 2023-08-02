import json
import os
import traceback

import torch
from safetensors.torch import load_file

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.util.TrainProgress import TrainProgress
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.enum.ModelType import ModelType


class StableDiffusionEmbeddingModelLoader(BaseModelLoader):
    def __init__(self):
        super(StableDiffusionEmbeddingModelLoader, self).__init__()

    @staticmethod
    def __load_ckpt(model: StableDiffusionModel, embedding_name: str):
        embedding_state = torch.load(embedding_name)

        string_to_param_dict = embedding_state['string_to_param']
        name = embedding_state['name']

        embedding = StableDiffusionModelEmbedding(
            name=name,
            vector=string_to_param_dict['*'],
            token_count=string_to_param_dict['*'].shape[0]
        )

        model.embeddings = [embedding]

    @staticmethod
    def __load_safetensors(model: StableDiffusionModel, embedding_name: str):
        embedding_state = load_file(embedding_name)

        tensor = embedding_state["emp_params"]

        embedding = StableDiffusionModelEmbedding(
            name="*",
            vector=tensor,
            token_count=tensor.shape[0]
        )

        model.embeddings = [embedding]

    @staticmethod
    def __load_internal(model: StableDiffusionModel, embedding_name: str):
        with open(os.path.join(embedding_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )

        # embedding model
        pt_embedding_name = os.path.join(embedding_name, "lora", "embedding.pt")
        safetensors_embedding_name = os.path.join(embedding_name, "lora", "embedding.safetensors")
        if os.path.exists(pt_embedding_name):
            StableDiffusionEmbeddingModelLoader.__load_ckpt(model, pt_embedding_name)
        elif os.path.exists(safetensors_embedding_name):
            StableDiffusionEmbeddingModelLoader.__load_safetensors(model, safetensors_embedding_name)

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
                self.__load_internal(model, extra_model_name)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_safetensors(model, extra_model_name)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_ckpt(model, extra_model_name)
                return model
            except:
                stacktraces.append(traceback.format_exc())
        else:
            return model

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + extra_model_name)
