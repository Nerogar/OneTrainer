import json
import os
import traceback

import torch
from safetensors.torch import load_file

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress
from modules.util.enum.ModelType import ModelType


class StableDiffusionXLEmbeddingModelLoader(BaseModelLoader):
    def __init__(self):
        super(StableDiffusionXLEmbeddingModelLoader, self).__init__()

    @staticmethod
    def __load_ckpt(model: StableDiffusionXLModel, embedding_name: str):
        embedding_state = torch.load(embedding_name)

        text_encoder_1_vector = embedding_state['clip_l']
        text_encoder_2_vector = embedding_state['clip_g']
        name = embedding_state['name']

        embedding = StableDiffusionXLModelEmbedding(
            name=name,
            text_encoder_1_vector=text_encoder_1_vector,
            text_encoder_2_vector=text_encoder_2_vector,
            token_count=text_encoder_1_vector.shape[0],
        )

        model.embeddings = [embedding]

    @staticmethod
    def __load_safetensors(model: StableDiffusionXLModel, embedding_name: str):
        embedding_state = load_file(embedding_name)

        text_encoder_1_vector = embedding_state['clip_l']
        text_encoder_2_vector = embedding_state['clip_g']

        embedding = StableDiffusionXLModelEmbedding(
            name="*",
            text_encoder_1_vector=text_encoder_1_vector,
            text_encoder_2_vector=text_encoder_2_vector,
            token_count=text_encoder_1_vector.shape[0],
        )

        model.embeddings = [embedding]

    @staticmethod
    def __load_internal(model: StableDiffusionXLModel, embedding_name: str):
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
            StableDiffusionXLEmbeddingModelLoader.__load_ckpt(model, pt_embedding_name)
        elif os.path.exists(safetensors_embedding_name):
            StableDiffusionXLEmbeddingModelLoader.__load_safetensors(model, safetensors_embedding_name)
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

    def load(
            self,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str | None,
            extra_model_name: str | None
    ) -> StableDiffusionXLModel | None:
        stacktraces = []

        base_model_loader = StableDiffusionXLModelLoader()

        if base_model_name is not None:
            model = base_model_loader.load(model_type, weight_dtypes, base_model_name, None)
        else:
            model = StableDiffusionXLModel(model_type=model_type)

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
