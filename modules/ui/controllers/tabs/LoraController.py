from modules.ui.controllers.BaseController import BaseController
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType

import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class LoraController(BaseController):
    state_ui_connections = {
        "peft_type": "typeCmb",
        "lora_model_name": "baseModelLed",
        "lora_rank": "rankSbx",
        "lora_alpha": "alphaSbx",
        "lora_decompose": "decomposeCbx",
        "lora_decompose_norm_epsilon": "normCbx",
        "lora_decompose_output_axis": "outputAxisCbx",
        "lora_weight_dtype": "weightDTypeCmb",
        "bundle_additional_embeddings": "bundleCbx",
        "oft_block_size": "oftBlockSizeSbx",
        "oft_coft": "coftCbx",
        "coft_eps": "coftLed",
        "oft_block_share": "blockShareCbx",
    }

    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/lora.ui", name=QCA.translate("main_window_tabs", "Lora"), parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.baseModelBtn, self.ui.baseModelLed, is_dir=False, save=False,
                               title=QCA.translate("dialog_window", "Open LoRA/LoHA/OFT 2 base model"),
                               filters=QCA.translate("filetype_filters", "Safetensors (*.safetensors);;Diffusers (model_index.json);;Checkpoints (*.ckpt *.pt *.bin);;All Files (*.*)"))

        self._connect([QtW.QApplication.instance().stateChanged, self.ui.typeCmb.activated],
                      self.__updateType(), update_after_connect=True)

        self._connect([QtW.QApplication.instance().stateChanged, self.ui.decomposeCbx.toggled],
                      self.__updateDora(), update_after_connect=True)

    def _connectInputValidation(self):
        # Alpha cannot be higher than rank.
        self._connect(self.ui.rankSbx.valueChanged, lambda x: (self.ui.alphaSbx.setMaximum(x)))


    def _loadPresets(self):
        for e in PeftType.enabled_values():
            self.ui.typeCmb.addItem(e.pretty_print(), userData=e)

        for e in DataType.enabled_values(context="lora"):
            self.ui.weightDTypeCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __updateDora(self):
        @Slot()
        def f():
            enabled = self.ui.decomposeCbx.isChecked()
            self.ui.normCbx.setEnabled(enabled)
            self.ui.outputAxisCbx.setEnabled(enabled)
        return f

    def __updateType(self):
        @Slot()
        def f():
            self.ui.doraFrm.setVisible(self.ui.typeCmb.currentData() == PeftType.LORA)
            self.ui.oftFrm.setVisible(self.ui.typeCmb.currentData() == PeftType.OFT_2)

            self.ui.oftBlockSizeLbl.setVisible(self.ui.typeCmb.currentData() == PeftType.OFT_2)
            self.ui.oftBlockSizeSbx.setVisible(self.ui.typeCmb.currentData() == PeftType.OFT_2)

            self.ui.rankLbl.setVisible(self.ui.typeCmb.currentData() != PeftType.OFT_2)
            self.ui.rankSbx.setVisible(self.ui.typeCmb.currentData() != PeftType.OFT_2)

            self.ui.alphaLbl.setVisible(self.ui.typeCmb.currentData() != PeftType.OFT_2)
            self.ui.alphaSbx.setVisible(self.ui.typeCmb.currentData() != PeftType.OFT_2)
        return f
