from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig


class StableDiffusionFineTuneSetup(
    BaseStableDiffusionSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionFineTuneSetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> Iterable[Parameter]:
        params = list()

        if config.text_encoder.train:
            params += list(model.text_encoder.parameters())

        if config.train_any_embedding():
            params += list(model.embedding_wrapper.parameters())

        if config.unet.train:
            params += list(model.unet.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if config.text_encoder.train:
            param_groups.append(
                self.create_param_groups(
                    config,
                    model.text_encoder.parameters(),
                    config.text_encoder.learning_rate,
                )
            )

        if config.train_any_embedding():
            param_groups.append(
                self.create_param_groups(
                    config,
                    model.embedding_wrapper.parameters(),
                    config.embedding_learning_rate,
                )
            )

        if config.unet.train:
            param_groups.append(
                self.create_param_groups(config, model.unet.parameters(), config.unet.learning_rate)
            )

        return param_groups

    def __setup_requires_grad(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        train_text_encoder = config.text_encoder.train and \
                             not self.stop_text_encoder_training_elapsed(config, model.train_progress)
        model.text_encoder.requires_grad_(train_text_encoder)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = embedding_config.train and \
                              not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)
            embedding.text_encoder_vector.requires_grad_(train_embedding)

        train_unet = config.unet.train and \
                     not self.stop_unet_training_elapsed(config, model.train_progress)
        model.unet.requires_grad_(train_unet)

        model.vae.requires_grad_(False)


    def setup_model(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()
        elif config.force_v_prediction:
            model.force_v_prediction()
        elif config.force_epsilon_prediction:
            model.force_epsilon_prediction()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, config), model.optimizer_state_dict, config
        )
        model.optimizer_state_dict = None

        model.ema = create.create_ema(
            self.create_parameters(model, config), model.ema_state_dict, config
        )
        model.ema_state_dict = None

        self._setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop
        text_encoder_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)
        model.depth_estimator_to(self.temp_device)

        if config.text_encoder.train:
            model.text_encoder.train()
        else:
            model.text_encoder.eval()

        model.vae.eval()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)

    def report_learning_rates(
            self,
            model,
            config,
            scheduler,
            tensorboard
    ):
        lrs = scheduler.get_last_lr()
        names = []
        if config.text_encoder.train:
            names.append("te")
        if config.train_any_embedding():
            names.append("embeddings")
        if config.unet.train:
            names.append("unet")
        assert len(lrs) == len(names)

        lrs = config.optimizer.optimizer.maybe_adjust_lrs(lrs, model.optimizer)

        for name, lr in zip(names, lrs):
            tensorboard.add_scalar(
                f"lr/{name}", lr, model.train_progress.global_step
            )
