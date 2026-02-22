from modules.model.ChromaModel import ChromaModel
from modules.modelSaver.chroma.ChromaEmbeddingSaver import ChromaEmbeddingSaver
from modules.modelSaver.chroma.ChromaModelSaver import ChromaModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

ChromaFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.CHROMA_1,
    model_class=ChromaModel,
    model_saver_class=ChromaModelSaver,
    embedding_saver_class=ChromaEmbeddingSaver,
)
