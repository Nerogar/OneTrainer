import os

from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.ImageModel import ImageModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.ImageMegapixels import ImageMegapixels
from modules.util.enum.ImageOptimization import ImageOptimization

import PySide6.QtGui as QtGui
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class ImageController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/image.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.directoryBtn, self.ui.directoryLed, is_dir=True, save=False, title=
        QCA.translate("dialog_window", "Open Dataset directory"))

        state_ui_connections = {
            "directory": "directoryLed",
            "verify_images": "verifyCbx",
            "sequential_rename": "renameCbx",
            "process_alpha": "replaceColorCbx",
            "resize_large_images": "resizeCbx",
            "resize_megapixels": "resizeCmb",
            "alpha_bg_color": "colorLed",
            "optimization_type": "optimizationCmb",
            "resize_custom_megapixels": "customSbx",
        }

        self._connectStateUI(state_ui_connections, ImageModel.instance(), update_after_connect=True)
        self._connect(self.ui.processBtn.clicked, self.__startProcessFiles())
        self._connect(self.ui.cancelBtn.clicked, self.__stopProcessFiles())
        self._connect(self.ui.resizeCmb.activated, self.__enableCustom(), update_after_connect=True)

        self.__enableControls(True)()

    def _loadPresets(self):
        for e in ImageOptimization.enabled_values():
            self.ui.optimizationCmb.addItem(e.pretty_print(), userData=e)

        for e in ImageMegapixels.enabled_values():
            self.ui.resizeCmb.addItem(e.pretty_print(), userData=e)


    def _connectInputValidation(self):
        self.ui.colorLed.setValidator(QtGui.QRegularExpressionValidator("random|-1|#[0-9a-f]{6}|[a-z]+", self.ui))

    ###Reactions###

    def __startProcessFiles(self):
        @Slot()
        def f():
            self.ui.statusTed.setPlainText("")

            if self.ui.directoryLed.text() != "":
                if os.path.isdir(self.ui.directoryLed.text()):
                    worker, name = WorkerPool.instance().createNamed(self.__processFiles(), "process_images", abort_flag=ImageModel.instance().abort_flag, inject_progress_callback=True)
                    if worker is not None:
                        worker.connectCallbacks(init_fn=self.__enableControls(False), result_fn=None,
                                       finished_fn=self.__enableControls(True),
                                       errored_fn=self.__enableControls(True), aborted_fn=self.__enableControls(True),
                                       progress_fn=self.__updateStatus())
                        WorkerPool.instance().start(name)
                else:
                    self._openAlert(QCA.translate("image_window", "Invalid Folder"),
                                    QCA.translate("image_window", "The selected input folder does not exist"),
                                    type="critical")
            else:
                self._openAlert(QCA.translate("image_window", "No Folder Selected"),
                                QCA.translate("image_window", "Please select an input folder"))

        return f

    def __stopProcessFiles(self):
        @Slot()
        def f():
            ImageModel.instance().abort_flag.set()
        return f

    def __enableControls(self, enabled):
        @Slot()
        def f():
            self.ui.processBtn.setEnabled(enabled)
            self.ui.cancelBtn.setEnabled(not enabled)
            if enabled:
                self.ui.progressBar.setValue(0)
        return f

    def __enableCustom(self):
        @Slot()
        def f():
            self.ui.customLbl.setEnabled(self.ui.resizeCmb.currentData() == ImageMegapixels.CUSTOM)
            self.ui.customSbx.setEnabled(self.ui.resizeCmb.currentData() == ImageMegapixels.CUSTOM)
        return f

    ###Utils###

    def __processFiles(self):
        def f(progress_fn=None):
            return ImageModel.instance().process_files(progress_fn=progress_fn)
        return f

    def __updateStatus(self):
        progress_fn = self._updateProgress(self.ui.progressBar)
        def f(data):
            if "status" in data:
                self.ui.statusLbl.setText(data["status"])

            if "data" in data:
                self.ui.statusTed.setPlainText(self.ui.statusTed.toPlainText() + "\n" + data["data"])

            progress_fn(data)
        return f
