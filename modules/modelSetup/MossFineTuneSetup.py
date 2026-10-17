from modules.model.MossModel import MossModel
from modules.modelSetup.BaseMossSetup import BaseMossSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.MOSS, TrainingMethod.FINE_TUNE)
class MossFineTuneSetup(
    BaseMossSetup,
):
    def create_parameters(
            self,
            model: MossModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer, config.transformer,
            freeze=ModuleFilter.create(config), debug=config.debug_mode,
        )

        if config.text_encoder.train_embedding:
            for embedding, embedding_config in zip(model.all_embeddings(),
                                                    config.all_embedding_configs(), strict=True):
                if not embedding_config.train:
                    continue
                parameter_group_collection.add_group(NamedParameterGroup(
                    unique_name=f"embeddings/{embedding.uuid}",
                    display_name=f"embeddings/{embedding.placeholder}",
                    parameters=[model.adjoint_tokens[embedding.uuid]],
                    learning_rate=config.embedding_learning_rate,
                ))

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)


    def setup_model(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        self._setup_embeddings(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: MossModel,
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
            model: MossModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        self.__setup_requires_grad(model, config)
