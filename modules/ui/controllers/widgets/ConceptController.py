from modules.ui.controllers.BaseController import BaseController
from modules.ui.models.ConceptModel import ConceptModel

import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtW
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Slot


class ConceptController(BaseController):
    def __init__(self, loader, concept_window, idx, parent=None):
        self.concept_window = concept_window
        self.idx = idx
        super().__init__(loader, "modules/ui/views/widgets/concept.ui", name=None, parent=parent)

    ###FSM###

    def _connectUIBehavior(self):
        self._connect(self.ui.conceptBtn.clicked, self.__openConceptWindow())
        self._connect(self.ui.enableCbx.clicked, self.__enableConcept())

        self._connect(QtW.QApplication.instance().conceptsChanged, self.__updateConcept(), update_after_connect=True)

    ###Reactions###

    def __openConceptWindow(self):
        @Slot()
        def f():
            self._openWindow(self.concept_window, fixed_size=False)
            QtW.QApplication.instance().openConcept.emit(self.idx)
        return f

    def __enableConcept(self):
        @Slot()
        def f():
            ConceptModel.instance().set_state(f"{self.idx}.enabled", self.ui.enableCbx.isChecked())
            QtW.QApplication.instance().conceptsChanged.emit(True)
        return f

    def __updateConcept(self):
        @Slot()
        def f():
            self.ui.enableCbx.setChecked(ConceptModel.instance().get_state(f"{self.idx}.enabled"))
            self.ui.enableCbx.setText(ConceptModel.instance().get_concept_name(self.idx))

            img = ConceptModel.instance().get_preview_icon(self.idx)
            self.ui.conceptBtn.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(ImageQt(img))))
        return f
