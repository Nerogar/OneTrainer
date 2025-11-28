from modules.ui.controllers.BaseController import BaseController
from modules.ui.controllers.widgets.SampleController import SampleController
from modules.ui.controllers.windows.NewSampleController import NewSampleController
from modules.ui.controllers.windows.SampleController import SampleController as SampleControllerWindow
from modules.ui.models.SampleModel import SampleModel
from modules.ui.models.StateModel import StateModel
from modules.ui.models.TrainingModel import TrainingModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.TimeUnit import TimeUnit

import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class SamplingController(BaseController):
    state_ui_connections = {
        "sample_after": "sampleAfterSbx",
        "sample_after_unit": "sampleAfterCmb",
        "sample_skip_first": "skipSbx",
        "sample_image_format": "formatCmb",
        "non_ema_sampling": "nonEmaCbx",
        "samples_to_tensorboard": "tensorboardCbx",
    }

    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/sampling.ui", name=QCA.translate("main_window_tabs", "Sampling"), parent=parent)

    ###FSM###

    def _setup(self):
        self.children = []
        self.sample_params_window = NewSampleController(self.loader, parent=self)
        self.manual_sample_window = SampleControllerWindow(self.loader, parent=None)

    def _connectUIBehavior(self):
        self._connect(self.ui.addSampleBtn.clicked, self.__appendSample())
        self._connect(self.ui.toggleBtn.clicked, self.__toggleSamples())
        self._connect(self.ui.manualSampleBtn.clicked, self.__openSampleWindow())
        self._connect(self.ui.sampleNowBtn.clicked, self.__startSample())

        self._connect(QtW.QApplication.instance().stateChanged, self.__updateConfigs(), update_after_connect=True)

        self._connect([QtW.QApplication.instance().samplesChanged, QtW.QApplication.instance().stateChanged],
                      self.__updateSamples(), update_after_connect=True)

        self._connect([self.ui.configCmb.textActivated, QtW.QApplication.instance().stateChanged],
                      self.__loadConfig(), update_after_connect=True, initial_args=[self.ui.configCmb.currentText()])

        self._connect([QtW.QApplication.instance().aboutToQuit, QtW.QApplication.instance().samplesChanged],
                      self.__saveConfig())


    def _connectInputValidation(self):
        self.ui.configCmb.setValidator(QtGui.QRegularExpressionValidator(r"[a-zA-Z0-9_\-.][a-zA-Z0-9_\-. ]*", self.ui))


    def _loadPresets(self):
        for e in TimeUnit.enabled_values():
            self.ui.sampleAfterCmb.addItem(e.pretty_print(), userData=e)
        for e in ImageFormat.enabled_values():
            self.ui.formatCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __openSampleWindow(self):
        @Slot()
        def f():
            self._openWindow(self.manual_sample_window, fixed_size=True)
        return f

    def __loadConfig(self):
        def f(filename=None):
            if filename is None:
                filename = self.ui.configCmb.currentText()
            SampleModel.instance().load_config(filename)
            QtW.QApplication.instance().samplesChanged.emit()
        return f

    def __saveConfig(self):
        @Slot()
        def f():
            SampleModel.instance().save_config()
        return f

    def __updateConfigs(self):
        @Slot()
        def f():
            configs = SampleModel.instance().load_available_config_names("training_samples", include_default=False)
            if len(configs) == 0:
                configs.append(("samples", "training_samples/samples.json"))

            self.ui.configCmb.clear()
            for k, v in configs:
                self.ui.configCmb.addItem(k, userData=v)

            self.ui.configCmb.setCurrentIndex(self.ui.configCmb.findData(StateModel.instance().get_state("sample_definition_file_name")))
        return f

    def __updateSamples(self):
        @Slot()
        def f():
            for c in self.children:
                c._disconnectAll()

            self.ui.listWidget.clear()
            self.children = []

            for idx, _ in enumerate(SampleModel.instance().get_state("")):
               wdg = SampleController(self.loader, self.sample_params_window, idx, parent=self)
               self.children.append(wdg)
               self._appendWidget(self.ui.listWidget, wdg, self_delete_fn=self.__deleteSample(idx), self_clone_fn=self.__cloneSample(idx))

            if SampleModel.instance().some_samples_enabled():
                self.ui.toggleBtn.setText(QCA.translate("main_window_tabs", "Disable All"))
            else:
                self.ui.toggleBtn.setText(QCA.translate("main_window_tabs", "Enable All"))

        return f

    def __appendSample(self):
        @Slot()
        def f():
            SampleModel.instance().create_new_sample()
            QtW.QApplication.instance().samplesChanged.emit()
        return f

    def __toggleSamples(self):
        @Slot()
        def f():
            SampleModel.instance().toggle_samples()
            QtW.QApplication.instance().samplesChanged.emit()
        return f

    def __cloneSample(self, idx):
        @Slot()
        def f():
            SampleModel.instance().clone_sample(idx)
            QtW.QApplication.instance().samplesChanged.emit()

        return f

    def __deleteSample(self, idx):
        @Slot()
        def f():
            SampleModel.instance().delete_sample(idx)
            QtW.QApplication.instance().samplesChanged.emit()

        return f

    def __startSample(self):
        @Slot()
        def f():
            worker, name = WorkerPool.instance().createNamed(self.__sampleNow(), "sampling_operations", poolless=True, daemon=True,
                                                             inject_progress_callback=True)
            if worker is not None:
                worker.connectCallbacks(init_fn=self.__enableControls(False), result_fn=None,
                               finished_fn=self.__enableControls(True))
                WorkerPool.instance().start(name)

        return f

    def __enableControls(self, enabled):
        @Slot()
        def f():
            self.ui.sampleNowBtn.setEnabled(enabled)
        return f

    ###Utils###

    def __sampleNow(self):
        def f(progress_fn=None):
            TrainingModel.instance().sample_now()
        return f
