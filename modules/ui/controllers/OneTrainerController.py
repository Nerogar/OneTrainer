from modules.ui.controllers.BaseController import BaseController
from modules.ui.controllers.tabs.AdditionalEmbeddingsController import AdditionalEmbeddingsController
from modules.ui.controllers.tabs.BackupController import BackupController
from modules.ui.controllers.tabs.CloudController import CloudController
from modules.ui.controllers.tabs.ConceptsController import ConceptsController
from modules.ui.controllers.tabs.DataController import DataController
from modules.ui.controllers.tabs.EmbeddingsController import EmbeddingsController
from modules.ui.controllers.tabs.GeneralController import GeneralController
from modules.ui.controllers.tabs.LoraController import LoraController
from modules.ui.controllers.tabs.ModelController import ModelController
from modules.ui.controllers.tabs.SamplingController import SamplingController
from modules.ui.controllers.tabs.ToolsController import ToolsController
from modules.ui.controllers.tabs.TrainingController import TrainingController
from modules.ui.controllers.windows.SaveController import SaveController
from modules.ui.models.BulkModel import BulkModel
from modules.ui.models.CaptionModel import CaptionModel
from modules.ui.models.ImageModel import ImageModel
from modules.ui.models.MaskModel import MaskModel
from modules.ui.models.StateModel import StateModel
from modules.ui.models.TrainingModel import TrainingModel
from modules.ui.utils.WorkerPool import WorkerPool
from modules.util.enum.ModelFlags import ModelFlags
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import PySide6.QtGui as QtG
import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


# Main window.
class OnetrainerController(BaseController):
    state_ui_connections = {
        "model_type": "modelTypeCmb",
        "training_method": "trainingTypeCmb"
    }

    def __init__(self, loader):
        super().__init__(loader, "modules/ui/views/windows/onetrainer.ui", name="OneTrainer", parent=None)

    ###FSM###

    def _setup(self):
        self.save_window = SaveController(self.loader, parent=self)
        self.children = {}
        self.__createTabs()
        self.training = False

    def _connectUIBehavior(self):
        self._connect(self.ui.wikiBtn.clicked, lambda: self._openUrl("https://github.com/Nerogar/OneTrainer/wiki"))
        self._connect(self.ui.saveConfigBtn.clicked, lambda: self._openWindow(self.save_window, fixed_size=True))
        self._connect(self.ui.exportBtn.clicked, lambda: self.__exportConfig())
        self._connect(self.ui.startBtn.clicked, self.__toggleTrain())
        self._connect(self.ui.debugBtn.clicked, self.__startDebug())
        self._connect(self.ui.tensorboardBtn.clicked, self.__openTensorboard())

        self._connect([self.ui.trainingTypeCmb.activated, self.ui.modelTypeCmb.activated, QtW.QApplication.instance().stateChanged],
                      self.__changeModel(), update_after_connect=True)

        self._connect(self.ui.configCmb.activated, lambda idx: self.__loadConfig(self.ui.configCmb.currentData(), idx))

        self._connect([self.ui.modelTypeCmb.activated, QtW.QApplication.instance().stateChanged],
                      self.__updateModel(), update_after_connect=True)

        self._connect(QtW.QApplication.instance().stateChanged, self.__updateConfigs(), update_after_connect=True)
        self._connect(QtW.QApplication.instance().savedConfig, self.__updateSelectedConfig())

        self._connect(QtW.QApplication.instance().aboutToQuit, self.__onQuit())

        self.__loadConfig("training_presets/#.json")  # Load last config.
        QtW.QApplication.instance().stateChanged.emit()

    def _loadPresets(self):
        for e in ModelType.enabled_values(context="main_window"):
            self.ui.modelTypeCmb.addItem(e.pretty_print(), userData=e)

    ###Reactions###

    def __openTensorboard(self):
        @Slot()
        def f():
            self._openUrl("http://localhost:" + str(StateModel.instance().get_state("tensorboard_port")))
        return f

    def __startDebug(self):
        @Slot()
        def f():
            diag = QtW.QFileDialog()
            txt, _ = diag.getSaveFileName(parent=None,
                                            dir="OneTrainer_debug_report.zip",
                                            caption=QCA.translate("main_window", "Save Debug Package"),
                                            filter=QCA.translate("filetype_filters", "Zip (*.zip)"))
            if txt != "":
                worker, name = WorkerPool.instance().createNamed(self.__generate_debug_package(txt), "generate_debug", poolless=True, inject_progress_callback=True)
                if worker is not None:
                    worker.connectCallbacks(init_fn=self.__enableDebugControls(False), result_fn=None,
                                   finished_fn=self.__enableDebugControls(True),
                                   errored_fn=self.__enableDebugControls(True),
                                   progress_fn=self.__updateStatus())
                    WorkerPool.instance().start(name)
        return f

    def __enableDebugControls(self, enabled):
        @Slot()
        def f():
            self.ui.debugBtn.setEnabled(enabled)
        return f

    def __updateSelectedConfig(self):
        @Slot(str)
        def f(config):
            self.ui.configCmb.setCurrentText(config)
        return f

    def __updateConfigs(self):
        @Slot()
        def f():
            configs = StateModel.instance().load_available_config_names("training_presets")
            self.ui.configCmb.clear()
            self.save_window.ui.configCmb.clear()
            for k, v in configs:
                self.ui.configCmb.addItem(k, userData=v)
                if not k.startswith("#"):
                    self.save_window.ui.configCmb.addItem(k, userData=v)
        return f

    def __updateModel(self):
        @Slot()
        def f():
            flags = ModelFlags.getFlags(self.ui.modelTypeCmb.currentData(), self.ui.trainingTypeCmb.currentData())

            old_training_type = self.ui.trainingTypeCmb.currentData()

            self.ui.trainingTypeCmb.clear()

            self.ui.trainingTypeCmb.addItem(QCA.translate("training_method", "Fine Tune"), userData=TrainingMethod.FINE_TUNE)
            self.ui.trainingTypeCmb.addItem(QCA.translate("training_method", "LoRA"), userData=TrainingMethod.LORA)

            if ModelFlags.CAN_TRAIN_EMBEDDING in flags:
                self.ui.trainingTypeCmb.addItem(QCA.translate("training_method", "Embedding"), userData=TrainingMethod.EMBEDDING)
            if ModelFlags.CAN_FINE_TUNE_VAE in flags:
                self.ui.trainingTypeCmb.addItem(QCA.translate("training_method", "Fine Tune VAE"), userData=TrainingMethod.FINE_TUNE_VAE)

            if old_training_type is not None:
                self.ui.trainingTypeCmb.setCurrentIndex(self.ui.trainingTypeCmb.findData(old_training_type))
                self.ui.trainingTypeCmb.activated.emit(self.ui.trainingTypeCmb.findData(old_training_type))
            else:
                old_training_type = StateModel.instance().get_state("training_method")
                if old_training_type is not None:
                    self.ui.trainingTypeCmb.setCurrentIndex(self.ui.trainingTypeCmb.findData(old_training_type))
                else:
                    self.ui.trainingTypeCmb.activated.emit(0)

        return f

    def __enableControls(self, state):
        @Slot()
        def f():
            if state == "enabled": # Startup and successful termination.
                self.training = False
                self.ui.startBtn.setEnabled(True)
                self.ui.startBtn.setText(QCA.translate("main_window", "Start Training"))
                self.ui.startBtn.setPalette(self.ui.palette())
                self.ui.stepPrg.setValue(0)
                self.ui.epochPrg.setValue(0)
                self.ui.etaLbl.setText("")
            elif state == "running":
                self.training = True
                self.ui.startBtn.setEnabled(True)
                self.ui.startBtn.setText(QCA.translate("main_window", "Stop Training"))
                self.ui.startBtn.setPalette(QtG.QPalette(QtG.QColor("green")))
            elif state == "stopping":
                self.training = True
                self.ui.startBtn.setEnabled(False)
                self.ui.startBtn.setText(QCA.translate("main_window", "Stopping..."))
                self.ui.startBtn.setPalette(QtG.QPalette(QtG.QColor("red")))
            elif state == "cancelled": # Interrupted or errored termination. Do not update progress bars, as we might be interested in knowing in which epoch/step the error occurred.
                self.training = False
                self.ui.startBtn.setText(QCA.translate("main_window", "Start Training"))
                self.ui.startBtn.setPalette(QtG.QPalette(QtG.QColor("darkred")))
                self.ui.startBtn.setEnabled(True)
                self.ui.etaLbl.setText("")
        return f

    def __updateStatus(self):
        @Slot(dict)
        def f(data):
            if "status" in data:
                self.ui.statusLbl.setText(data["status"])

            if "eta" in data:
                self.ui.etaLbl.setText(f"ETA: {data['eta']}")

            if "step" in data and "max_steps" in data:
                val = int(data["step"] / data["max_steps"] * self.ui.stepPrg.maximum()) if data["max_steps"] > 0 else 0
                self.ui.stepPrg.setValue(val)
            if "epoch" in data and "max_epochs" in data:
                val = int(data["epoch"] / data["max_epochs"] * self.ui.epochPrg.maximum()) if data["max_epochs"] > 0 else 0
                self.ui.epochPrg.setValue(val)

            if "event" in data:
                self.__enableControls(data["event"])()

        return f

    def __changeModel(self):
        @Slot()
        def f():
            model_type = self.ui.modelTypeCmb.currentData()
            training_type = self.ui.trainingTypeCmb.currentData()
            self.ui.tabWidget.setTabVisible(self.children["lora"]["index"], training_type == TrainingMethod.LORA)
            self.ui.tabWidget.setTabVisible(self.children["embedding"]["index"], training_type == TrainingMethod.EMBEDDING)

            QtW.QApplication.instance().modelChanged.emit(model_type, training_type)
        return f

    def __onQuit(self):
        @Slot()
        def f():
            StateModel.instance().save_default()
            StateModel.instance().stop_tensorboard()
            CaptionModel.instance().release_model()
            MaskModel.instance().release_model()
            ImageModel.instance().terminate_pool()
            BulkModel.instance().terminate_pool()
        return f

    def __toggleTrain(self):
        @Slot()
        def f():
            if self.training:
                self.__stopTrain()
            else:
                worker, name = WorkerPool.instance().createNamed(self.__train(), "train", poolless=True, daemon=True, inject_progress_callback=True)
                if worker is not None:
                    worker.connectCallbacks(init_fn=self.__enableControls("running"), result_fn=None,
                                   finished_fn=self.__enableControls("enabled"),
                                   errored_fn=self.__enableControls("cancelled"), aborted_fn=self.__enableControls("cancelled"),
                                   progress_fn=self.__updateStatus())
                    WorkerPool.instance().start(name)

        return f

    ###Utils###

    def __generate_debug_package(self, zip_path):
        def f(progress_fn=None):
            StateModel.instance().generate_debug_package(zip_path, progress_fn=progress_fn)

        return f

    def __createTabs(self):
        for name, controller in [
            ("general", GeneralController),
            ("model", ModelController),
            ("data", DataController),
            ("concepts", ConceptsController),
            ("training", TrainingController),
            ("sampling", SamplingController),
            ("backup", BackupController),
            ("tools", ToolsController),
            ("additional_embeddings", AdditionalEmbeddingsController),
            ("cloud", CloudController),
            ("lora", LoraController),
            ("embedding", EmbeddingsController)
        ]:
            c = controller(self.loader, parent=self)
            self.children[name] = {"controller": c, "index": len(self.children)}
            self.ui.tabWidget.addTab(c.ui, c.name)

        self.ui.tabWidget.setTabVisible(self.children["lora"]["index"], False)
        self.ui.tabWidget.setTabVisible(self.children["embedding"]["index"], False)

    def __loadConfig(self, config, idx=None):
        StateModel.instance().load_config(config)
        QtW.QApplication.instance().stateChanged.emit()
        QtW.QApplication.instance().embeddingsChanged.emit()
        if idx is not None:
            self.ui.configCmb.setCurrentIndex(idx)


    def __exportConfig(self):
        diag = QtW.QFileDialog()
        txt, flt = diag.getSaveFileName(parent=None, caption=QCA.translate("dialog_window", "Save Config"), dir=None,
                                        filter=QCA.translate("filetype_filters", "JSON (*.json)"))
        if txt != "":
            filename = self._appendExtension(txt, flt)
            StateModel.instance().save_config(filename)


    def __train(self):
        def f(progress_fn=None):
            TrainingModel.instance().train(progress_fn=progress_fn)
        return f

    def __stopTrain(self):
        TrainingModel.instance().stop_training()
