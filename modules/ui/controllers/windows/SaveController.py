from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.StateModel import StateModel

import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtW
from PySide6.QtCore import Slot


class SaveController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/save.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connect(self.ui.cancelBtn.clicked, lambda: self.ui.hide())
        self._connect(self.ui.okBtn.clicked, self.__save())

    def _connectInputValidation(self):
        self.ui.configCmb.setValidator(QtGui.QRegularExpressionValidator(r"[a-zA-Z0-9_\-.][a-zA-Z0-9_\-. ]*", self.ui))

    ###Reactions###

    def __save(self):
        @Slot()
        def f():
            name = self.ui.configCmb.currentText()
            if name != "" and not name.startswith("#"):
                StateModel.instance().save_to_file(name)

                QtW.QApplication.instance().savedConfig.emit(name)
                self.ui.hide()
        return f
