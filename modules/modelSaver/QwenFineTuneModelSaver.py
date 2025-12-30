from modules.model.QwenModel import QwenModel
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.modelSaver.qwen.QwenModelSaver import QwenModelSaver
from modules.util.enum.ModelType import ModelType

QwenFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.QWEN,
    model_class=QwenModel,
    model_saver_class=QwenModelSaver,
    embedding_saver_class=None,
)
