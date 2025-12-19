from modules.model.SanaModel import SanaModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.sana.SanaEmbeddingSaver import SanaEmbeddingSaver
from modules.modelSaver.sana.SanaModelSaver import SanaModelSaver
from modules.util.enum.ModelType import ModelType

SanaFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.SANA,
    model_class=SanaModel,
    model_saver_class=SanaModelSaver,
    embedding_saver_class=SanaEmbeddingSaver,
)
