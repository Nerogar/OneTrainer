from modules.model.IdeogramModel import IdeogramModel
from modules.modelSetup.BaseIdeogramSetup import BaseIdeogramSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.IDEOGRAM_4, TrainingMethod.LORA)
class IdeogramLoRASetup(
    BaseIdeogramSetup,
):
    def create_parameters(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()
        self._create_model_part_parameters(parameter_group_collection, "transformer", model.transformer_lora, config.transformer)
        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        if model.unconditional_transformer is not None:
            model.unconditional_transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(
            self,
            model: IdeogramModel,
            config: TrainConfig,
    ):
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(","),
            fusion_spec=model.fusion_groups(), fuse=config.output_model_format.needs_qkv_fusion(),
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
            model: IdeogramModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        parts = ["transformer"]
        if text_encoder_on_train_device:
            parts.append("text_encoder")
        if vae_on_train_device:
            parts.append("vae")
        # the unconditional transformer is only needed for sampling; materialize_only() evicts it as it's
        # not in parts, keeping it off the train device during training
        model.materialize_only(*parts)

        model.text_encoder.eval()
        model.vae.eval()
        if model.unconditional_transformer is not None:
            model.unconditional_transformer.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: IdeogramModel,
            config: TrainConfig,
            train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)
