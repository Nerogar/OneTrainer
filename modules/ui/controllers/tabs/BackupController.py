from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.TrainingModel import TrainingModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.TimeUnit import TimeUnit

import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class BackupController(BaseController):
    state_ui_connections = {
        "backup_after": "backupSbx",
        "backup_after_unit": "backupCmb",
        "rolling_backup": "rollingBackupCbx",
        "backup_before_save": "backupBeforeSaveCbx",
        "rolling_backup_count": "rollingCountSbx",
        "save_every": "saveSbx",
        "save_every_unit": "saveCmb",
        "save_skip_first": "skipSbx",
        "save_filename_prefix": "savePrefixLed"
    }

    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/backup.ui", name=QCA.translate("main_window_tabs", "Backup"), parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connect(self.ui.backupBtn.clicked, self.__startBackup())
        self._connect(self.ui.saveBtn.clicked, self.__startSave())

        self._connect([QtW.QApplication.instance().stateChanged, self.ui.backupCmb.activated],
                      self.__updateBackup(), update_after_connect=True)

        self._connect([QtW.QApplication.instance().stateChanged, self.ui.saveCmb.activated],
                      self.__updateSave(), update_after_connect=True)

        self._connect([QtW.QApplication.instance().stateChanged, self.ui.rollingBackupCbx.toggled],
                      self.__updateRollingBackup(), update_after_connect=True)

    def _loadPresets(self):
        for e in TimeUnit.enabled_values():
            self.ui.backupCmb.addItem(e.pretty_print(), userData=e)
        for e in TimeUnit.enabled_values():
            self.ui.saveCmb.addItem(e.pretty_print(), userData=e)


    ###Reactions###

    def __startBackup(self):
        @Slot()
        def f():
            worker, name = WorkerPool.instance().createNamed(self.__backupNow(), "backup_operations", poolless=True, daemon=True,
                                                             inject_progress_callback=True)
            if worker is not None:
                worker.connectCallbacks(init_fn=self.__enableControls(False), result_fn=None,
                               finished_fn=self.__enableControls(True))
                WorkerPool.instance().start(name)
        return f

    def __startSave(self):
        @Slot()
        def f():
            worker, name = WorkerPool.instance().createNamed(self.__saveNow(), "backup_operations", poolless=True, daemon=True,
                                                             inject_progress_callback=True)
            if worker is not None:
                worker.connectCallbacks(init_fn=self.__enableControls(False), result_fn=None,
                               finished_fn=self.__enableControls(True))
                WorkerPool.instance().start(name)

        return f

    def __enableControls(self, enabled):
        @Slot()
        def f():
            self.ui.backupBtn.setEnabled(enabled)
            self.ui.saveBtn.setEnabled(enabled)
        return f

    def __updateBackup(self):
        @Slot()
        def f():
            enabled = self.ui.backupCmb.currentData() != TimeUnit.NEVER

            self.ui.backupSbx.setEnabled(enabled)
            self.ui.rollingBackupCbx.setEnabled(enabled)
            self.ui.rollingCountSbx.setEnabled(enabled and self.ui.rollingBackupCbx.isChecked())
            self.ui.rollingCountLbl.setEnabled(enabled and self.ui.rollingBackupCbx.isChecked())

        return f

    def __updateRollingBackup(self):
        @Slot()
        def f():
            enabled = self.ui.rollingBackupCbx.isChecked()
            self.ui.rollingCountSbx.setEnabled(enabled and self.ui.backupCmb.currentData() != TimeUnit.NEVER)
            self.ui.rollingCountLbl.setEnabled(enabled and self.ui.backupCmb.currentData() != TimeUnit.NEVER)

        return f

    def __updateSave(self):
        @Slot()
        def f():
            enabled = self.ui.saveCmb.currentData() != TimeUnit.NEVER

            self.ui.saveSbx.setEnabled(enabled)
            self.ui.skipSbx.setEnabled(enabled)
            self.ui.rollingCountSbx.setEnabled(enabled)
            self.ui.savePrefixLed.setEnabled(enabled)
            self.ui.skipLbl.setEnabled(enabled)
            self.ui.savePrefixLbl.setEnabled(enabled)

        return f

    ###Utils###

    def __backupNow(self):
        def f(progress_fn=None):
            TrainingModel.instance().backup_now()
        return f

    def __saveNow(self):
        def f(progress_fn=None):
            TrainingModel.instance().save_now()
        return f
