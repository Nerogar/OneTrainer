import torch

from modules.model.PixArtAlphaModel import PixArtAlphaModel
from modules.modelSetup.BasePixArtAlphaSetup import BasePixArtAlphaSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters


class PixArtAlphaLoRASetup(
    BasePixArtAlphaSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(PixArtAlphaLoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_lora",
                display_name="text_encoder_lora",
                parameters=model.text_encoder_lora.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.train_any_embedding():
            for parameter, placeholder, name in zip(model.embedding_wrapper.additional_embeddings,
                                                    model.embedding_wrapper.additional_embedding_placeholders,
                                                    model.embedding_wrapper.additional_embedding_names):
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings/{name}",
                    display_name=f"embeddings/{placeholder}",
                    parameters=[parameter],
                    learning_rate=config.embedding_learning_rate,
                ))

        if config.prior.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer_lora",
                display_name="transformer_lora",
                parameters=model.transformer_lora.parameters(),
                learning_rate=config.prior.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_lora is not None:
            train_text_encoder = config.text_encoder.train and \
                                 not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_lora.requires_grad_(train_text_encoder)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = embedding_config.train and \
                              not self.stop_additional_embedding_training_elapsed(embedding_config,
                                                                                  model.train_progress, i)
            embedding.text_encoder_vector.requires_grad_(train_embedding)

        if model.transformer_lora is not None:
            train_prior = config.prior.train and \
                          not self.stop_prior_training_elapsed(config, model.train_progress)
            model.transformer_lora.requires_grad_(train_prior)

    def setup_model(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        model.text_encoder_lora = LoRAModuleWrapper(
            model.text_encoder, config.lora_rank, "lora_te", config.lora_alpha
        )

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, config.lora_rank, "lora_transformer", config.lora_alpha, ["attn1", "attn2"]
        )

        if model.lora_state_dict:
            model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.text_encoder_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.set_dropout(config.dropout_probability)

        model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())

        model.text_encoder_lora.hook_to_module()
        model.transformer_lora.hook_to_module()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config))

        self.setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: PixArtAlphaModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or config.align_prop or not config.latent_caching
        text_encoder_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

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
            model: PixArtAlphaModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
