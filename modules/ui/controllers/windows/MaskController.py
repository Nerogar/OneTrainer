import os

from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.MaskModel import MaskModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.GenerateMasksModel import GenerateMasksAction, GenerateMasksModel

from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class MaskController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/generate_mask.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.folderBtn, self.ui.folderLed, is_dir=True, save=False, title=
                               QCA.translate("dialog_window", "Open Dataset directory"))

        state_ui_connections = {
            "model": "modelCmb",
            "path": "folderLed",
            "prompt": "promptLed",
            "mode": "modeCmb",
            "alpha": "alphaSbx",
            "threshold": "thresholdSbx",
            "smooth": "smoothSbx",
            "expand": "expandSbx",
            "include_subdirectories": "includeSubfolderCbx"
        }

        self._connectStateUI(state_ui_connections, MaskModel.instance(), update_after_connect=True)

        self.__enableControls(True)()

        self._connect(self.ui.createMaskBtn.clicked, self.__startMask())

    def _loadPresets(self):
        for e in GenerateMasksModel.enabled_values():
            self.ui.modelCmb.addItem(e.pretty_print(), userData=e)

        for e in GenerateMasksAction.enabled_values():
            self.ui.modeCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __startMask(self):
        @Slot()
        def f():
            if self.ui.folderLed.text() != "":
                if os.path.isdir(self.ui.folderLed.text()):
                    worker, name = WorkerPool.instance().createNamed(self.__createMask(), "create_mask", inject_progress_callback=True)
                    if worker is not None:
                        worker.connectCallbacks(init_fn=self.__enableControls(False), result_fn=None,
                                       finished_fn=self.__enableControls(True),
                                       errored_fn=self.__enableControls(True), aborted_fn=self.__enableControls(True),
                                       progress_fn=self._updateProgress(self.ui.progressBar))
                        WorkerPool.instance().start(name)
                else:
                    self._openAlert(QCA.translate("mask_window", "Invalid Folder"),
                                    QCA.translate("mask_window", "The selected input folder does not exist"),
                                    type="critical")
            else:
                self._openAlert(QCA.translate("mask_window", "No Folder Selected"),
                                QCA.translate("mask_window", "Please select an input folder"))

        return f

    def __enableControls(self, enabled):
        @Slot()
        def f():
            self.ui.createMaskBtn.setEnabled(enabled)
            if enabled:
                self.ui.progressBar.setValue(0)
        return f


    ###Utils###

    def __createMask(self):
        def f(progress_fn=None):
            return MaskModel.instance().create_masks(progress_fn=progress_fn)

        return f
