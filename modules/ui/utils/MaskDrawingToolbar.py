from modules.util.enum.EditMode import EditMode
from modules.util.enum.ToolType import ToolType

import PySide6.QtGui as QtG
import PySide6.QtWidgets as QtW
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt


# Toolbar class for the FigureWidget.
class MaskDrawingToolbar(NavigationToolbar):
    toolitems = []  # Override default matplotlib tools.

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent, coordinates=False)
        self.tools = {}
        self.mode = EditMode.NONE

    def toggleTool(self, tool_mode):
        if self.canvas.widgetlock.available(self):
            if self.mode == tool_mode:
                self.mode = EditMode.NONE
                self.canvas.widgetlock.release(self)
            else:
                self.mode = tool_mode
                self.canvas.widgetlock(self)

            for k, v in self.tools.items():
                v.setChecked(k == tool_mode)

    def addTools(self, tools):
        for t in tools:
            self.__addTool(t)

    def __addTool(self, tool):
        if tool["type"] == ToolType.SEPARATOR:
            self.addSeparator()
        elif tool["type"] == ToolType.SPINBOX or tool["type"] == ToolType.DOUBLE_SPINBOX:
            range = tool.get("spinbox_range", (0.05, 1.0, 0.05, 0.05))
            value = tool.get("value", range[0])
            wdg = QtW.QLabel(self.canvas, text=tool.get("text", None))
            if tool["type"] == ToolType.DOUBLE_SPINBOX:
                wdg2 = QtW.QDoubleSpinBox(self.canvas, objectName=tool.get("name", None))
            else:
                wdg2 = QtW.QSpinBox(self.canvas, objectName=tool.get("name", None))
            wdg2.setMinimum(range[0])
            wdg2.setMaximum(range[1])
            wdg2.setSingleStep(range[2])
            wdg2.setValue(value)
            if "fn" in tool:
                wdg2.valueChanged.connect(tool["fn"])
                tool["fn"](value)

            wdg.setBuddy(wdg2)
            if "icon" in tool:
                wdg.setPixmap(QtG.QPixmap(tool["icon"]))
            if "tooltip" in tool:
                wdg2.setToolTip(tool["tooltip"])
            self.addWidget(wdg)
            self.addWidget(wdg2)
        else:
            wdg = QtW.QToolButton(self.canvas, objectName=tool.get("name", None))
            if "shortcut" in tool:
                scut = QtG.QShortcut(QtG.QKeySequence(tool["shortcut"]), self.canvas)
                scut.setAutoRepeat(False)
                scut.activated.connect(wdg.click)

            if tool["type"] == ToolType.CHECKABLE_BUTTON:
                wdg.setCheckable(True)
                self.tools[tool["tool"]] = wdg
                wdg.clicked.connect(lambda: self.toggleTool(tool["tool"]))
            elif "fn" in tool:
                wdg.clicked.connect(tool["fn"])

            if "text" in tool and "icon" in tool:
                wdg.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

            if "text" in tool:
                wdg.setText(tool["text"])

            if "icon" in tool:
                wdg.setIcon(QtG.QIcon(tool["icon"]))

            if "tooltip" in tool:
                wdg.setToolTip(tool["tooltip"])

            self.addWidget(wdg)
