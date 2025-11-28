from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.StateModel import StateModel

from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class ProfileController(BaseController):
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/windows/profile.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connect(self.ui.dumpBtn.clicked, self.__dump())
        self._connect(self.ui.startBtn.clicked, self.__toggleProfiling())

    ###Reactions###

    def __dump(self):
        @Slot()
        def f():
            StateModel.instance().dump_stack()
        return f

    def __toggleProfiling(self):
        @Slot()
        def f():
            StateModel.instance().toggle_profiler()
            if StateModel.instance().is_profiling:
                self.ui.statusLbl.setText(QCA.translate("profiling_window", "Profiling active..."))
                self.ui.startBtn.setText(QCA.translate("profiling_window", "End Profiling"))
            else:
                self.ui.statusLbl.setText(QCA.translate("profiling_window", "Inactive"))
                self.ui.startBtn.setText(QCA.translate("profiling_window", "Start Profiling"))

            # TODO: this button exits the application if not run from Scalene. It would be nice to disable it when running from python.
            #  However the library does not expose a function to check before running scalene_profiler.start()
        return f
