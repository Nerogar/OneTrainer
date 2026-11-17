from math import ceil

from modules.util.ui.UIState import BaseUIState

from PySide6.QtCore import QEvent, QMetaObject, Qt
from PySide6.QtGui import QPalette
from PySide6.QtQml import QQmlEngine, QQmlProperty
from PySide6.QtQuickWidgets import QQuickWidget


class QtQuickSwitch(QQuickWidget):
    """A real Qt Quick Controls Switch bound to the existing Widgets UI state."""

    def __init__(self, parent, ui_state: BaseUIState, name: str, label: str):
        super().__init__(parent)
        self.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        # GeneralTabView supplies the enclosing panel color before display.
        # An opaque matching surface avoids Quick/Widgets alpha-compositing
        # artifacts on hardware renderers without altering the switch style.
        self.setClearColor(parent.palette().color(QPalette.ColorRole.Window))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(label)
        self.setInitialProperties({"focus": True, "objectName": label})
        self.loadFromModule("QtQuick.Controls", "Switch")
        if self.status() != QQuickWidget.Status.Ready:
            raise RuntimeError("Could not load Qt Quick Switch: " + "; ".join(error.toString() for error in self.errors()))
        root = self.rootObject()
        # Resolve the attached accessibility property in the built-in control's
        # QML context, replacing the Accessible.name binding in the old file.
        QQmlProperty(root, "Accessible.name", QQmlEngine.contextForObject(root)).write(label)
        self.setFixedSize(ceil(root.implicitWidth()), ceil(root.implicitHeight()))
        var = ui_state.get_var(name)
        self.setChecked(bool(var.get()))
        # toggled is emitted for mouse, keyboard and drag interaction, not for
        # programmatic checked updates, so preset loading cannot feed back.
        root.toggled.connect(lambda: var.set(self.isChecked()))
        binding = var._bind_widget(self.setChecked)
        self.destroyed.connect(lambda: var._unbind_widget(binding))

    def isChecked(self) -> bool:
        return bool(self.rootObject().property("checked"))

    def setChecked(self, checked):
        self.rootObject().setProperty("checked", bool(checked))

    def click(self):
        if self.isEnabled():
            QMetaObject.invokeMethod(self.rootObject(), "click")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange and self.rootObject() is not None:
            self.rootObject().setProperty("enabled", self.isEnabled())
