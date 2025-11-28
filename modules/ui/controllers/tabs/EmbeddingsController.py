from modules.ui.controllers.BaseController import BaseController
from modules.util.enum.DataType import DataType

from PySide6.QtCore import QCoreApplication as QCA


class EmbeddingsController(BaseController):
    state_ui_connections = {
        "embedding.model_name": "baseEmbeddingLed",
        "embedding.token_count": "tokenSbx",
        "embedding.initial_embedding_text": "initialEmbeddingLed",
        "embedding_weight_dtype": "embeddingDTypeCmb",
        "embedding.placeholder": "placeholderLed",
        "embedding.is_output_embedding": "outputCbx",
    }

    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/embeddings.ui", name=QCA.translate("main_window_tabs", "Embeddings"), parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.baseEmbeddingBtn, self.ui.baseEmbeddingLed, is_dir=False, save=False,
                               title=QCA.translate("dialog_window", "Open base embeddings"),
                               filters=QCA.translate("filetype_filters", "Safetensors (*.safetensors);;Diffusers (model_index.json);;Checkpoints (*.ckpt, *.pt, *.bin);;All Files (*.*)")) # TODO: Maybe refactor filters in ENUM?

    def _loadPresets(self):
        for e in DataType.enabled_values(context="embeddings"):
            self.ui.embeddingDTypeCmb.addItem(e.pretty_print(), userData=e)
