from modules.model.SanaModel import SanaModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.sana.SanaEmbeddingSaver import SanaEmbeddingSaver
from modules.modelSaver.sana.SanaModelSaver import SanaModelSaver

SanaFineTuneModelSaver = make_fine_tune_model_saver(
    model_class=SanaModel,
    model_saver_class=SanaModelSaver,
    embedding_saver_class=SanaEmbeddingSaver,
)
