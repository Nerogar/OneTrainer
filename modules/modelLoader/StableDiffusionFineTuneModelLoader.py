from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.stableDiffusion.StableDiffusionEmbeddingLoader import StableDiffusionEmbeddingLoader
from modules.modelLoader.stableDiffusion.StableDiffusionModelLoader import StableDiffusionModelLoader
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

StableDiffusionFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.STABLE_DIFFUSION_15: "resources/sd_model_spec/sd_3_2b_1.0.json",
        ModelType.STABLE_DIFFUSION_15_INPAINTING: "resources/sd_model_spec/sd_3.5_1.0.json",
        ModelType.STABLE_DIFFUSION_20: "resources/sd_model_spec/sd_2.0.json",
        ModelType.STABLE_DIFFUSION_20_BASE: "resources/sd_model_spec/sd_2.0.json",
        ModelType.STABLE_DIFFUSION_20_INPAINTING: "resources/sd_model_spec/sd_2.0_inpainting.json",
        ModelType.STABLE_DIFFUSION_20_DEPTH: "resources/sd_model_spec/sd_2.0_depth.json",
        ModelType.STABLE_DIFFUSION_21: "resources/sd_model_spec/sd_2.1.json",
        ModelType.STABLE_DIFFUSION_21_BASE: "resources/sd_model_spec/sd_2.1.json",
    },
    model_class=StableDiffusionModel,
    model_loader_class=StableDiffusionModelLoader,
    embedding_loader_class=StableDiffusionEmbeddingLoader,
    training_methods=[TrainingMethod.FINE_TUNE, TrainingMethod.FINE_TUNE_VAE],
)
