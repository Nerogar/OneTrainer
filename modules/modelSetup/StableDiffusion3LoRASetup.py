import torch

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSetup.BaseStableDiffusion3Setup import BaseStableDiffusion3Setup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.NamedParameterGroup import NamedParameterGroupCollection, NamedParameterGroup
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix


class StableDiffusion3LoRASetup(
    BaseStableDiffusion3Setup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusion3LoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        if config.text_encoder.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_1",
                display_name="text_encoder_1",
                parameters=model.text_encoder_1_lora.parameters(),
                learning_rate=config.text_encoder.learning_rate,
            ))

        if config.text_encoder_2.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_2",
                display_name="text_encoder_2",
                parameters=model.text_encoder_2_lora.parameters(),
                learning_rate=config.text_encoder_2.learning_rate,
            ))

        if config.text_encoder_3.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="text_encoder_3",
                display_name="text_encoder_3",
                parameters=model.text_encoder_3_lora.parameters(),
                learning_rate=config.text_encoder_3.learning_rate,
            ))

        if config.train_any_embedding():
            for parameter, placeholder, name in zip(model.embedding_wrapper_1.additional_embeddings,
                                                    model.embedding_wrapper_1.additional_embedding_placeholders,
                                                    model.embedding_wrapper_1.additional_embedding_names):
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings_1/{name}",
                    display_name=f"embeddings_1/{placeholder}",
                    parameters=[parameter],
                    learning_rate=config.embedding_learning_rate,
                ))

            for parameter, placeholder, name in zip(model.embedding_wrapper_2.additional_embeddings,
                                                    model.embedding_wrapper_2.additional_embedding_placeholders,
                                                    model.embedding_wrapper_2.additional_embedding_names):
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings_2/{name}",
                    display_name=f"embeddings_2/{placeholder}",
                    parameters=[parameter],
                    learning_rate=config.embedding_learning_rate,
                ))

            for parameter, placeholder, name in zip(model.embedding_wrapper_3.additional_embeddings,
                                                    model.embedding_wrapper_3.additional_embedding_placeholders,
                                                    model.embedding_wrapper_3.additional_embedding_names):
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings_3/{name}",
                    display_name=f"embeddings_3/{placeholder}",
                    parameters=[parameter],
                    learning_rate=config.embedding_learning_rate,
                ))

        if config.prior.train:
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name="transformer",
                display_name="transformer",
                parameters=model.transformer_lora.parameters(),
                learning_rate=config.prior.learning_rate,
            ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
    ):

        if model.text_encoder_1 is not None:
            model.text_encoder_1.requires_grad_(False)
        if model.text_encoder_2 is not None:
            model.text_encoder_2.requires_grad_(False)
        if model.text_encoder_3 is not None:
            model.text_encoder_3.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = config.text_encoder.train and \
                                   not self.stop_text_encoder_training_elapsed(config, model.train_progress)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)

        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = config.text_encoder_2.train and \
                                   not self.stop_text_encoder_2_training_elapsed(config, model.train_progress)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)

        if model.text_encoder_3_lora is not None:
            train_text_encoder_3 = config.text_encoder_3.train and \
                                   not self.stop_text_encoder_3_training_elapsed(config, model.train_progress)
            model.text_encoder_3_lora.requires_grad_(train_text_encoder_3)

        for i, embedding in enumerate(model.additional_embeddings):
            embedding_config = config.additional_embeddings[i]
            train_embedding = embedding_config.train and \
                              not self.stop_additional_embedding_training_elapsed(embedding_config, model.train_progress, i)

            if embedding.text_encoder_1_vector is not None:
                embedding.text_encoder_1_vector.requires_grad_(train_embedding)
            if embedding.text_encoder_2_vector is not None:
                embedding.text_encoder_2_vector.requires_grad_(train_embedding)
            if embedding.text_encoder_3_vector is not None:
                embedding.text_encoder_3_vector.requires_grad_(train_embedding)

        if model.transformer_lora is not None:
            train_transformer = config.prior.train and \
                         not self.stop_prior_training_elapsed(config, model.train_progress)
            model.transformer_lora.requires_grad_(train_transformer)


    def setup_model(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
    ):
        create_te1 = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te1")
        create_te2 = config.text_encoder_2.train or state_dict_has_prefix(model.lora_state_dict, "lora_te2")
        create_te3 = config.text_encoder_3.train or state_dict_has_prefix(model.lora_state_dict, "lora_te3")

        if model.text_encoder_1 is not None:
            model.text_encoder_1_lora = LoRAModuleWrapper(
                model.text_encoder_1, config.lora_rank, "lora_te1", config.lora_alpha
            ) if create_te1 else None

        if model.text_encoder_2 is not None:
            model.text_encoder_2_lora = LoRAModuleWrapper(
                model.text_encoder_2, config.lora_rank, "lora_te2", config.lora_alpha
            ) if create_te2 else None

        if model.text_encoder_3 is not None:
            model.text_encoder_3_lora = LoRAModuleWrapper(
                model.text_encoder_3, config.lora_rank, "lora_te3", config.lora_alpha
            ) if create_te3 else None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, config.lora_rank, "lora_transformer", config.lora_alpha, ["attn"]
        )

        if model.lora_state_dict:
            if model.text_encoder_1_lora is not None:
                model.text_encoder_1_lora.load_state_dict(model.lora_state_dict)
            if model.text_encoder_2_lora is not None:
                model.text_encoder_2_lora.load_state_dict(model.lora_state_dict)
            if model.text_encoder_3_lora is not None:
                model.text_encoder_3_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.set_dropout(config.dropout_probability)
            model.text_encoder_1_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_1_lora.hook_to_module()

        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.set_dropout(config.dropout_probability)
            model.text_encoder_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_2_lora.hook_to_module()

        if model.text_encoder_3_lora is not None:
            model.text_encoder_3_lora.set_dropout(config.dropout_probability)
            model.text_encoder_3_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_3_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        if config.train_any_embedding():
            if model.text_encoder_1 is not None:
                model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            if model.text_encoder_2 is not None:
                model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            if model.text_encoder_3 is not None:
                model.text_encoder_3.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_3)
        self._setup_additional_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config))

        self._setup_optimizations(model, config)

    def setup_train_device(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
    ):
        vae_on_train_device = config.align_prop or not config.latent_caching
        text_encoder_1_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        text_encoder_2_on_train_device = \
            config.text_encoder_2.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        text_encoder_3_on_train_device = \
            config.text_encoder_3.train \
            or config.train_any_embedding() \
            or config.align_prop \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.text_encoder_3_to(self.train_device if text_encoder_3_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder_1:
            if config.text_encoder.train:
                model.text_encoder_1.train()
            else:
                model.text_encoder_1.eval()

        if model.text_encoder_2:
            if config.text_encoder_2.train:
                model.text_encoder_2.train()
            else:
                model.text_encoder_2.eval()

        if model.text_encoder_3:
            if config.text_encoder_3.train:
                model.text_encoder_3.train()
            else:
                model.text_encoder_3.eval()

        model.vae.eval()

        if config.prior.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusion3Model,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            if model.embedding_wrapper_1 is not None:
                model.embedding_wrapper_1.normalize_embeddings()
            if model.embedding_wrapper_2 is not None:
                model.embedding_wrapper_2.normalize_embeddings()
            if model.embedding_wrapper_3 is not None:
                model.embedding_wrapper_3.normalize_embeddings()
        self.__setup_requires_grad(model, config)
