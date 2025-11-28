from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.StateModel import StateModel
from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.TimeUnit import TimeUnit

import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class GeneralController(BaseController):
    state_ui_connections = {
        "workspace_dir": "workspaceLed",
        "continue_last_backup": "continueCbx",
        "debug_mode": "debugCbx",
        "tensorboard": "tensorboardCbx",
        "tensorboard_expose": "exposeTensorboardCbx",
        "validation": "validateCbx",
        "dataloader_threads": "dataloaderSbx",
        "train_device": "trainDeviceLed",
        "multi_gpu": "multiGpuCbx",
        "gradient_reduce_precision": "gradientReduceCmb",
        "async_gradient_reduce": "asyncGradientCbx",
        "temp_device": "tempDeviceLed",
        "cache_dir": "cacheLed",
        "only_cache": "onlyCacheCbx",
        "debug_dir": "debugLed",
        "tensorboard_always_on": "alwaysOnTensorboardCbx",
        "tensorboard_port": "tensorboardSbx",
        "validate_after": "validateSbx",
        "validate_after_unit": "validateCmb",
        "device_indexes": "deviceIndexesLed",
        "fused_gradient_reduce": "fusedGradientCbx",
        "async_gradient_reduce_buffer": "bufferSbx",
    }

    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/general.ui", name=QCA.translate("main_window_tabs", "General"), parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connectFileDialog(self.ui.workspaceBtn, self.ui.workspaceLed, is_dir=True, save=False,
                               title=QCA.translate("dialog_window", "Open Workspace directory"))
        self._connectFileDialog(self.ui.cacheBtn, self.ui.cacheLed, is_dir=True, save=False,
                               title=QCA.translate("dialog_window", "Open Cache directory"))
        self._connectFileDialog(self.ui.debugBtn, self.ui.debugLed, is_dir=True, save=False,
                               title=QCA.translate("dialog_window", "Open Debug directory"))

        self._connect([self.ui.alwaysOnTensorboardCbx.toggled, self.ui.workspaceLed.editingFinished, QtW.QApplication.instance().stateChanged],
                      self.__toggleTensorboard(), update_after_connect=True)

        self._connect([self.ui.validateCbx.toggled, QtW.QApplication.instance().stateChanged],
                      self.__updateValidate(), update_after_connect=True)

        self._connect([self.ui.tensorboardCbx.toggled, QtW.QApplication.instance().stateChanged],
                      self.__updateTensorboard(), update_after_connect=True)

        self._connect([self.ui.debugCbx.toggled, QtW.QApplication.instance().stateChanged],
                      self.__updateDebug(), update_after_connect=True)

    def _loadPresets(self):
        for e in GradientReducePrecision.enabled_values():
            self.ui.gradientReduceCmb.addItem(e.pretty_print(), userData=e)

        for e in TimeUnit.enabled_values():
            self.ui.validateCmb.addItem(e.pretty_print(), userData=e)

    def _connectInputValidation(self):
        self.ui.deviceIndexesLed.setValidator(QtGui.QRegularExpressionValidator(r"(\d+(,\d+)*)?", self.ui))

        # TODO: trainDeviceLed and tempDeviceLed may be restricted to a comma-separated list of available torch devices.
        # However it is not possible to get a list of all possible devices without some hackish error handling, according to https://github.com/pytorch/pytorch/issues/97026
        # torch.testing.get_all_device_types() returns the *current* machine's devices, which may be unsuitable for cloud training.


    ###Reactions###

    def __updateValidate(self):
        @Slot()
        def f():
            enabled = self.ui.validateCbx.isChecked()
            self.ui.validateLbl.setEnabled(enabled)
            self.ui.validateSbx.setEnabled(enabled)
            self.ui.validateCmb.setEnabled(enabled)
        return f

    def __updateTensorboard(self):
        @Slot()
        def f():
            enabled = self.ui.tensorboardCbx.isChecked()
            self.ui.alwaysOnTensorboardCbx.setEnabled(enabled)
            self.ui.exposeTensorboardCbx.setEnabled(enabled)
            self.ui.tensorboardLbl.setEnabled(enabled)
            self.ui.tensorboardSbx.setEnabled(enabled)
        return f

    def __updateDebug(self):
        @Slot()
        def f():
            enabled = self.ui.debugCbx.isChecked()
            self.ui.debugLbl.setEnabled(enabled)
            self.ui.debugLed.setEnabled(enabled)
            self.ui.debugBtn.setEnabled(enabled)
        return f

    def __toggleTensorboard(self):
        @Slot()
        def f():
            if self.ui.alwaysOnTensorboardCbx.isChecked():
                StateModel.instance().start_tensorboard()
            else:
                StateModel.instance().stop_tensorboard()
        return f
