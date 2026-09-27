from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.WUERSTCHEN_2, TrainingMethod.EMBEDDING)
@factory.register(BaseModelSetup, ModelType.STABLE_CASCADE_1, TrainingMethod.EMBEDDING)
class WuerstchenEmbeddingSetup(
    BaseWuerstchenSetup,
):
    def create_parameters(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._add_embedding_param_groups(
            model.all_prior_text_encoder_embeddings(), parameter_group_collection, config.embedding_learning_rate,
            "prior_embeddings"
        )

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        model.prior_text_encoder.requires_grad_(False)
        model.prior_prior.requires_grad_(False)
        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.requires_grad_(False)
        model.decoder_decoder.requires_grad_(False)
        model.decoder_vqgan.requires_grad_(False)
        model.effnet_encoder.requires_grad_(False)

    def setup_model(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        model.prior_text_encoder.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        effnet_on_train_device = not config.latent_caching

        # decoder/decoder_text_encoder/decoder_vqgan are never needed during prior training; materialize_only()
        # evicts them (decoder_text_encoder only exists in model_parts() for Wuerstchen v2, not Stable Cascade)
        parts = ["prior", "text_encoder"]
        if effnet_on_train_device:
            parts.append("effnet_encoder")
        model.materialize_only(*parts)

        if model.model_type.is_wuerstchen_v2():
            model.decoder_text_encoder.eval()
        model.decoder_decoder.eval()
        model.decoder_vqgan.eval()
        model.effnet_encoder.eval()

        model.prior_text_encoder.eval()
        model.prior_prior.eval()

    def after_optimizer_step(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_prior_text_encoder_embeddings())
            model.prior_embedding_wrapper.normalize_embeddings()
        self.__setup_requires_grad(model, config)
