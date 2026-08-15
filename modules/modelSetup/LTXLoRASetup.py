from modules.model.LTXModel import LTXModel
from modules.modelSetup.BaseLTXSetup import BaseLTXSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.LTX_2, TrainingMethod.LORA)
class LTXLoRASetup(
    BaseLTXSetup,
):
    def create_parameters(
            self,
            model: LTXModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(parameter_group_collection, "transformer", model.transformer_lora, config.transformer)
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: LTXModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.connectors.requires_grad_(False)
        model.transformer.requires_grad_(False)
        if model.low_noise_transformer is not None:
            model.low_noise_transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        model.audio_vae.requires_grad_(False)
        model.vocoder.requires_grad_(False)
        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(
            self,
            model: LTXModel,
            config: TrainConfig,
    ):
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: LTXModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        parts = ["transformer"]
        if text_encoder_on_train_device:
            # the connectors run alongside the TE in the dataloader, so they are needed exactly when it is
            parts.append("text_encoder")
            parts.append("connectors")
        if vae_on_train_device:
            parts.append("vae")
        model.materialize_only(*parts)
        # keep the VAE latent stats on the train device: predict() normalizes with them every step,
        # and .to(cuda) from an offloaded VAE would block-sync the stream each step.
        model.vae.latents_mean = model.vae.latents_mean.to(self.train_device)
        model.vae.latents_std = model.vae.latents_std.to(self.train_device)

        model.text_encoder.eval()
        model.connectors.eval()
        model.vae.eval()
        if model.low_noise_transformer is not None:
            # absent from `parts` above: only the sampler runs it, which materializes it on demand
            model.low_noise_transformer.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: LTXModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)
