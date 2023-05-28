import json
import os

import torch
from safetensors.torch import load_file

from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionEmbeddingModelLoader(BaseModelLoader):
    def __init__(self):
        super(StableDiffusionEmbeddingModelLoader, self).__init__()

    @staticmethod
    def __load_ckpt(model: StableDiffusionModel, embedding_name: str) -> bool:
        try:
            embedding_state = torch.load(embedding_name)

            string_to_param_dict = embedding_state['string_to_param']
            name = embedding_state['name']

            embedding = StableDiffusionModelEmbedding(
                name=name,
                vector=string_to_param_dict['*'],
                token_count=string_to_param_dict['*'].shape[0]
            )

            model.embeddings = [embedding]

            return True
        except:
            return False

    @staticmethod
    def __load_safetensors(model: StableDiffusionModel, embedding_name: str) -> bool:
        try:
            embedding_state = load_file(embedding_name)

            tensor = embedding_state["emp_params"]

            embedding = StableDiffusionModelEmbedding(
                name="*",
                vector=tensor,
                token_count=tensor.shape[0]
            )

            model.embeddings = [embedding]

            return True
        except:
            return False

    @staticmethod
    def __load_internal(model: StableDiffusionModel, embedding_name: str) -> bool:
        try:
            with open(os.path.join(embedding_name, "meta.json"), "r") as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )

            # embedding model
            loaded = StableDiffusionEmbeddingModelLoader.__load_ckpt(
                model,
                os.path.join(embedding_name, "embedding", "embedding.pt")
            )
            if not loaded:
                return False

            # optimizer
            try:
                model.optimizer_state_dict = torch.load(os.path.join(embedding_name, "optimizer", "optimizer.pt"))
            except FileNotFoundError:
                pass

            # meta
            model.train_progress = train_progress

            return True
        except:
            return False

    def load(self, model_type: ModelType, base_model_name: str, extra_model_name: str) -> StableDiffusionModel | None:
        base_model_loader = StableDiffusionModelLoader()
        model = base_model_loader.load(model_type, base_model_name, None)

        embedding_loaded = self.__load_internal(model, extra_model_name)

        if not embedding_loaded:
            embedding_loaded = self.__load_safetensors(model, extra_model_name)

        if not embedding_loaded:
            embedding_loaded = self.__load_ckpt(model, extra_model_name)

        return model
