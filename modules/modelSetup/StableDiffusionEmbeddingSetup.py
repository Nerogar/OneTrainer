from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.modelSetup.mixin.ModelSetupClipEmbeddingMixin import ModelSetupClipEmbeddingMixin
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionEmbeddingSetup(
    BaseStableDiffusionSetup,
    ModelSetupClipEmbeddingMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionEmbeddingSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> Iterable[Parameter]:
        return model.embedding_wrapper.parameters()

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        return [
            self.create_param_groups(
                config,
                model.embedding_wrapper.parameters(),
                config.learning_rate,
            )
        ]

    def __setup_requires_grad(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.unet.requires_grad_(False)

        model.embedding.text_encoder_vector.requires_grad_(True)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = embedding_config.train and \
                              not self.stop_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_vector.requires_grad_(train_embedding)

    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, config), model.optimizer_state_dict, config
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, config), model.ema_state_dict, config
        )
        del model.ema_state_dict

        self._setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop_loss

        model.text_encoder.to(self.train_device)
        model.vae.to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet.to(self.train_device)
        model.depth_estimator_to(self.temp_device)

        model.text_encoder.eval()
        model.vae.eval()
        model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)

    def report_learning_rates(
            self,
            model,
            config,
            scheduler,
            tensorboard
    ):
        lr = scheduler.get_last_lr()[0]
        lr = config.optimizer.optimizer.maybe_adjust_lrs([lr], model.optimizer)[0]
        tensorboard.add_scalar("lr/embedding", lr, model.train_progress.global_step)
