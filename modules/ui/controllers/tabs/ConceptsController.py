from modules.ui.controllers.BaseController import BaseController
from modules.ui.controllers.widgets.ConceptController import ConceptController as WidgetConceptController
from modules.ui.controllers.windows.ConceptController import ConceptController as WinConceptController
from modules.ui.models.ConceptModel import ConceptModel
from modules.ui.models.StateModel import StateModel
from modules.util.enum.ConceptType import ConceptType

import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtW
from PySide6.QtCore import QCoreApplication as QCA
from PySide6.QtCore import Slot


class ConceptsController(BaseController):
    children = []
    def __init__(self, loader, parent=None):
        super().__init__(loader, "modules/ui/views/tabs/concepts.ui", name=QCA.translate("main_window_tabs", "Concepts"), parent=parent)

    ###FSM###

    def _setup(self):
        self.concept_window = WinConceptController(self.loader, parent=self)

    def _connectUIBehavior(self):
        self._connect(self.ui.addConceptBtn.clicked, self.__appendConcept())
        self._connect(self.ui.toggleBtn.clicked, self.__toggleConcepts())
        self._connect(self.ui.clearBtn.clicked, self.__clearFilters())

        self._connect(QtW.QApplication.instance().stateChanged, self.__updateConfigs(), update_after_connect=True)

        self._connect([QtW.QApplication.instance().aboutToQuit, QtW.QApplication.instance().conceptsChanged],
                      self.__saveConfig())

        self._connect([self.ui.searchLed.textChanged, self.ui.typeCmb.activated, self.ui.showDisabledCbx.toggled, QtW.QApplication.instance().stateChanged],
                      lambda: QtW.QApplication.instance().conceptsChanged.emit(False))


        self._connect([QtW.QApplication.instance().conceptsChanged, QtW.QApplication.instance().stateChanged],
                      self.__updateConcepts(), update_after_connect=True)


        self._connect([self.ui.presetCmb.textActivated, QtW.QApplication.instance().stateChanged], self.__loadConfig(), update_after_connect=True)



    def _loadPresets(self):
        for e in ConceptType.enabled_values(context="all"):
            self.ui.typeCmb.addItem(e.pretty_print(), userData=e)

    def _connectInputValidation(self):
        self.ui.presetCmb.setValidator(QtGui.QRegularExpressionValidator(r"[a-zA-Z0-9_\-.][a-zA-Z0-9_\-. ]*", self.ui))

    ###Reactions###

    def __updateConfigs(self):
        @Slot()
        def f():
            configs = ConceptModel.instance().load_available_config_names("training_concepts", include_default=False)
            if len(configs) == 0:
                configs.append(("concepts", "training_concepts/concepts.json"))

            for c in self.children:
                c._disconnectAll()

            self.ui.presetCmb.clear()
            for k, v in configs:
                self.ui.presetCmb.addItem(k, userData=v)

            self.ui.presetCmb.setCurrentIndex(self.ui.presetCmb.findData(StateModel.instance().get_state("concept_file_name")))
        return f

    def __loadConfig(self):
        def f(filename=None):
            if filename is None:
                filename = self.ui.presetCmb.currentText()
            ConceptModel.instance().load_config(filename)
            QtW.QApplication.instance().conceptsChanged.emit(False)
        return f

    def __saveConfig(self):
        @Slot(bool)
        def f(save=True):
            if save:
                ConceptModel.instance().save_config()
        return f

    def __appendConcept(self):
        @Slot()
        def f():
            ConceptModel.instance().create_new_concept()
            QtW.QApplication.instance().conceptsChanged.emit(True)
        return f

    def __toggleConcepts(self):
        @Slot()
        def f():
            ConceptModel.instance().toggle_concepts()
            QtW.QApplication.instance().conceptsChanged.emit(True)
        return f

    def __clearFilters(self):
        def f():
            self.ui.searchLed.setText("")
            self.ui.typeCmb.setCurrentIndex(self.ui.typeCmb.findData(ConceptType.ALL))
            self.ui.showDisabledCbx.setChecked(True)

            QtW.QApplication.instance().conceptsChanged.emit(False)

        return f

    def __updateConcepts(self):
        @Slot(bool)
        def f(save=False):
            for c in self.children:
                c._disconnectAll()

            self.ui.listWidget.clear()
            self.children = []

            for idx, _ in ConceptModel.instance().get_filtered_concepts(self.ui.searchLed.text(), self.ui.typeCmb.currentData(), self.ui.showDisabledCbx.isChecked()):
                wdg = WidgetConceptController(self.loader, self.concept_window, idx, parent=self)
                self.children.append(wdg)
                self._appendWidget(self.ui.listWidget, wdg, self_delete_fn=self.__deleteConcept(idx), self_clone_fn=self.__cloneConcept(idx))

            if ConceptModel.instance().some_concepts_enabled():
                self.ui.toggleBtn.setText(QCA.translate("main_window_tabs", "Disable All"))
            else:
                self.ui.toggleBtn.setText(QCA.translate("main_window_tabs", "Enable All"))

        return f

    def __cloneConcept(self, idx):
        @Slot()
        def f():
            ConceptModel.instance().clone_concept(idx)
            QtW.QApplication.instance().conceptsChanged.emit(True)

        return f

    def __deleteConcept(self, idx):
        @Slot()
        def f():
            ConceptModel.instance().delete_concept(idx)
            QtW.QApplication.instance().conceptsChanged.emit(True)

        return f
