from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.StateModel import StateModel
from modules.util.enum.TimeUnit import TimeUnit

import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA


class EmbeddingController(BaseController):
    def __init__(self, loader, idx, parent=None):
        self.idx = idx
        super().__init__(loader, "modules/ui/views/widgets/embedding.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.baseEmbeddingBtn, self.ui.baseEmbeddingLed, is_dir=False, save=False,
                               title=QCA.translate("dialog_window", "Open base embeddings"),
                               filters=QCA.translate("filetype_filters",
                                                     "Safetensors (*.safetensors);;Diffusers (model_index.json);;Checkpoints (*.ckpt *.pt *.bin);;All Files (*.*)"))

        self.dynamic_state_ui_connections = {
            "additional_embeddings.{idx}.model_name": "baseEmbeddingLed",
            "additional_embeddings.{idx}.placeholder": "placeholderLed",
            "additional_embeddings.{idx}.token_count": "tokenSbx",
            "additional_embeddings.{idx}.train": "trainCbx",
            "additional_embeddings.{idx}.is_output_embedding": "outputEmbeddingCbx",
            "additional_embeddings.{idx}.stop_training_after": "stopTrainingSbx",
            "additional_embeddings.{idx}.stop_training_after_unit": "stopTrainingCmb",
            "additional_embeddings.{idx}.initial_embedding_text": "initialEmbeddingLed",
        }

        self._connectStateUI(self.dynamic_state_ui_connections, StateModel.instance(),
                             signal=QtW.QApplication.instance().embeddingsChanged, update_after_connect=True,
                             idx=self.idx)

    def _loadPresets(self):
        for e in TimeUnit.enabled_values():
            self.ui.stopTrainingCmb.addItem(e.pretty_print(), userData=e)
