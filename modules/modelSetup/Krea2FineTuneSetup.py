from modules.model.Krea2Model import Krea2Model
from modules.modelSetup.BaseKrea2Setup import BaseKrea2Setup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.KREA_2, TrainingMethod.FINE_TUNE)
class Krea2FineTuneSetup(
    BaseKrea2Setup,
):
    def create_parameters(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer, config.transformer,
            freeze=ModuleFilter.create(config), debug=config.debug_mode,
        )

        if config.train_any_embedding() or config.train_any_output_embedding():
            raise NotImplementedError("Embeddings not implemented for Krea 2")

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)


    def setup_model(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_on_train_device = not config.latent_caching

        parts = ["transformer"]
        if text_encoder_on_train_device:
            parts.append("text_encoder")
        if vae_on_train_device:
            parts.append("vae")
        model.materialize_only(*parts)

        model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: Krea2Model,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)
