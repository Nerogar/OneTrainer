
import PySide6.QtGui as QtG
import PySide6.QtWidgets as QtW


# Scientific Notation Line Edit Widget.
class SNLineEdit(QtW.QLineEdit):
    def __init__(self, contents=None, parent=None, objectName=None):
        super().__init__(contents, parent, objectName=objectName)

        #self.setValidator(QtG.QDoubleValidator(parent)) # QDoubleValidator does not work for locales which do not use the decimal point "."
        self.setValidator(QtG.QRegularExpressionValidator(r"-?inf|[+-]?\d+\.?\d*([eE][+-]?\d+)?", parent)) # A regular expression seems more reliable.
