from modules.model.ChromaModel import ChromaModel
from modules.modelSaver.chroma.ChromaEmbeddingSaver import ChromaEmbeddingSaver
from modules.modelSaver.chroma.ChromaModelSaver import ChromaModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver

ChromaFineTuneModelSaver = make_fine_tune_model_saver(
    model_class=ChromaModel,
    model_saver_class=ChromaModelSaver,
    embedding_saver_class=ChromaEmbeddingSaver,
)
