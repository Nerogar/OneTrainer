from modules.model.ChromaModel import ChromaModel
from modules.modelSaver.chroma.ChromaEmbeddingSaver import ChromaEmbeddingSaver
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver

ChromaEmbeddingModelSaver = make_embedding_model_saver(
    model_class=ChromaModel,
    embedding_saver_class=ChromaEmbeddingSaver,
)
