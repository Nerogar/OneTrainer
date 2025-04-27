from modules.model.SanaModel import SanaModel
from modules.modelSetup.BaseSanaSetup import BaseSanaSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch

PRESETS = {
    "attn-mlp": ["attn1", "attn2", "ff."],
    "attn-only": ["attn1", "attn2"],
    "full": [],
}


class SanaLoRASetup(
    BaseSanaSetup,
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
            model: SanaModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_lora",
                parameters=model.text_encoder_lora.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.train_any_embedding() or config.train_any_output_embedding():
            self._add_embedding_param_groups(
                model.all_text_encoder_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                "embeddings"
            )

        if config.prior.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer_lora",
                parameters=model.transformer_lora.parameters(),
                learning_rate=config.prior.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_lora is not None:
            train_text_encoder = config.text_encoder.train and \
                                 not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_lora.requires_grad_(train_text_encoder)

        if model.transformer_lora is not None:
            train_prior = config.prior.train and \
                          not self.stop_prior_training_elapsed(config, model.train_progress)
            model.transformer_lora.requires_grad_(train_prior)

    def setup_model(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te")
        model.text_encoder_lora = LoRAModuleWrapper(
            model.text_encoder, "lora_te", config
        ) if create_te else None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "lora_transformer", config, config.lora_layers.split(",")
        )

        if model.lora_state_dict:
            if create_te:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if config.text_encoder.train:
            model.text_encoder_lora.set_dropout(config.dropout_probability)
        if create_te:
            model.text_encoder_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: SanaModel,
            config: TrainConfig,
    ):
        vae_on_train_device = self.debug_mode or not config.latent_caching
        text_encoder_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
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
            model: SanaModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_embeddings())
            model.embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
