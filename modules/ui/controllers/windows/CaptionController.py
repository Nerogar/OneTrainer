import os

from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.CaptionModel import CaptionModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsAction, GenerateCaptionsModel

from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class CaptionController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/generate_caption.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.folderBtn, self.ui.folderLed, is_dir=True, save=False, title=
                               QCA.translate("dialog_window", "Open Dataset directory"))

        state_ui_connections = {
            "model": "modelCmb",
            "path": "folderLed",
            "caption": "initialCaptionLed",
            "prefix": "captionPrefixLed",
            "postfix": "captionPostfixLed",
            "mode": "modeCmb",
            "include_subdirectories": "includeSubfolderCbx"
        }

        self._connectStateUI(state_ui_connections, CaptionModel.instance(), update_after_connect=True)
        self._connect(self.ui.createMaskBtn.clicked, self.__startCaption())

        self.__enableControls(True)()

    def _loadPresets(self):
        for e in GenerateCaptionsModel.enabled_values():
            self.ui.modelCmb.addItem(e.pretty_print(), userData=e)

        for e in GenerateCaptionsAction.enabled_values():
            self.ui.modeCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __startCaption(self):
        @Slot()
        def f():
            if self.ui.folderLed.text() != "":
                if os.path.isdir(self.ui.folderLed.text()):
                    worker, name = WorkerPool.instance().createNamed(self.__createCaption(), "create_caption", inject_progress_callback=True)
                    if worker is not None:
                        worker.connectCallbacks(init_fn=self.__enableControls(False), result_fn=None,
                                       finished_fn=self.__enableControls(True),
                                       errored_fn=self.__enableControls(True), aborted_fn=self.__enableControls(True),
                                       progress_fn=self._updateProgress(self.ui.progressBar))
                        WorkerPool.instance().start(name)
                else:
                    self._openAlert(QCA.translate("caption_window", "Invalid Folder"),
                                    QCA.translate("caption_window", "The selected input folder does not exist"), type="critical")
            else:
                self._openAlert(QCA.translate("caption_window", "No Folder Selected"),
                                QCA.translate("caption_window", "Please select an input folder"))

        return f

    def __enableControls(self, enabled):
        @Slot()
        def f():
            self.ui.createMaskBtn.setEnabled(enabled)
            if enabled:
                self.ui.progressBar.setValue(0)
        return f

    ###Utils###

    def __createCaption(self):
        def f(progress_fn=None):
            return CaptionModel.instance().create_captions(progress_fn=progress_fn)

        return f
