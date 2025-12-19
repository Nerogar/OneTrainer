from modules.model.ChromaModel import ChromaModel
from modules.modelSaver.chroma.ChromaEmbeddingSaver import ChromaEmbeddingSaver
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.util.enum.ModelType import ModelType

ChromaEmbeddingModelSaver = make_embedding_model_saver(
    ModelType.CHROMA_1,
    model_class=ChromaModel,
    embedding_saver_class=ChromaEmbeddingSaver,
)
