from modules.model.ChromaModel import ChromaModel
from modules.modelSetup.BaseChromaSetup import BaseChromaSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class ChromaFineTuneSetup(
    BaseChromaSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: ChromaModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder",
                parameters=model.text_encoder.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings"
                )

        if config.prior.train:
            #TODO apply a configurable layer filter
            filtered_parameters = [param[1] for param in model.transformer.named_parameters() if "guidance_layer" not in param[0]]
            print("Warning: not training layers ", end='')
            print(', '.join([param[0] for param in model.transformer.named_parameters() if "guidance_layer" in param[0]]))
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer",
                parameters=filtered_parameters,
                learning_rate=config.prior.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: ChromaModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)

        if model.text_encoder is not None:
            train_text_encoder = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder.requires_grad_(train_text_encoder)

        train_transformer = config.prior.train and \
                     not self.stop_prior_training_elapsed(config, model.train_progress)
        model.transformer.requires_grad_(train_transformer)

        model.vae.requires_grad_(False)


    def setup_model(
            self,
            model: ChromaModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            if model.text_encoder is not None:
                model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: ChromaModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder:
            if config.text_encoder.train:
                model.text_encoder.train()
            else:
                model.text_encoder.eval()

        model.vae.eval()

        if config.prior.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: ChromaModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            if model.embedding_wrapper is not None:
                model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
