import threading
import traceback

from modules.ui.BaseGenerateMasksWindowView import BaseGenerateMasksWindowView
from modules.ui.GenerateMasksWindowController import GenerateMasksWindowController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QFileDialog, QLineEdit, QPushButton, QWidget


class PySide6GenerateMasksWindowView(BaseGenerateMasksWindowView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: GenerateMasksWindowController, path, parent_include_subdirectories):
        QDialog.__init__(self, parent)
        self.controller = controller
        self.is_running = False
        self._worker = None
        self.ui_state = PySide6UIState({
            "model": "ClipSeg",
            "mode": "Create if absent",
            "include_subdirectories": parent_include_subdirectories,
        })
        self.models = ["ClipSeg", "Rembg", "Rembg-Human", "Hex Color"]
        self.modes = ["Replace all masks", "Create if absent", "Add to existing", "Subtract from existing", "Blend with existing"]

        self.setWindowTitle("Batch generate masks")
        self.resize(360, 430)
        self.setModal(True)
        self.finished.connect(self.deleteLater)

        outer = pyside6_components._layout(self)
        pad = pyside6_components.PAD
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setColumnStretch(1, 1)
        outer.setRowStretch(3, 1)
        self.frame = QWidget(self)
        outer.addWidget(self.frame, 0, 0, 1, 2)
        layout = pyside6_components._layout(self.frame)
        layout.setColumnStretch(1, 1)

        pyside6_components.label(self.frame, 0, 0, "Model")
        self.model_dropdown = pyside6_components.options(self.frame, 0, 1, self.models, self.ui_state, "model")
        pyside6_components.label(self.frame, 1, 0, "Folder")
        path_frame = QWidget(self.frame)
        path_layout = pyside6_components._layout(path_frame)
        path_layout.setColumnStretch(0, 1)
        layout.addWidget(path_frame, 1, 1)
        self.path_entry = QLineEdit(path if path is not None else "", path_frame)
        self.path_entry.setMinimumWidth(140)
        self.path_entry.setAccessibleName("Folder")
        path_layout.addWidget(self.path_entry, 0, 0)
        pyside6_components.button(path_frame, 0, 1, "...", self.browse_for_path, width=30)

        self.prompt_entry = self._entry(2, "Prompt")
        pyside6_components.label(self.frame, 3, 0, "Mode")
        self.mode_dropdown = pyside6_components.options(self.frame, 3, 1, self.modes, self.ui_state, "mode")
        self.threshold_entry = self._entry(4, "Threshold", "0.3", "0.0 - 1.0")
        self.smooth_entry = self._entry(5, "Smooth", "5", "5")
        self.expand_entry = self._entry(6, "Expand", "10", "10")
        self.alpha_entry = self._entry(7, "Alpha", "1", "1")
        pyside6_components.label(self.frame, 8, 0, "Include subfolders")
        self.include_subdirectories_switch = pyside6_components.switch(
            self.frame, 8, 1, self.ui_state, "include_subdirectories",
        )

        self.progress_label = pyside6_components.label(self, 1, 0, "Progress: 0/0")
        self.progress = pyside6_components.progress(self, 1, 1)
        self.progress.setTextVisible(False)
        outer.setAlignment(self.progress, Qt.AlignVCenter)
        self.create_masks_button = QPushButton("Create Masks", self)
        self.create_masks_button.clicked.connect(self._on_create_masks)
        outer.addWidget(self.create_masks_button, 2, 0, 1, 2)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

    def _entry(self, row, label, text="", placeholder=""):
        pyside6_components.label(self.frame, row, 0, label)
        entry = QLineEdit(text, self.frame)
        entry.setMinimumWidth(140)
        entry.setPlaceholderText(placeholder)
        pyside6_components._add(pyside6_components._layout(self.frame), entry, row, 1)
        return entry

    def browse_for_path(self):
        self.path_entry.setText(QFileDialog.getExistingDirectory(self))

    def set_progress(self, value, max_value):
        progress = int(value / max_value * 1000)
        QTimer.singleShot(0, self, lambda: self._update_progress(value, max_value, progress))

    def _update_progress(self, value, max_value, progress):
        self.progress.setValue(progress)
        self.progress_label.setText(f"{value}/{max_value}")

    def _on_create_masks(self):
        if self.is_running:
            return
        arguments = {
            "model_name": self.model_dropdown.currentText(),
            "path": self.path_entry.text(),
            "prompt": self.prompt_entry.text(),
            "mode_str": self.mode_dropdown.currentText(),
            "alpha_str": self.alpha_entry.text(),
            "threshold_str": self.threshold_entry.text(),
            "smooth_str": self.smooth_entry.text(),
            "expand_str": self.expand_entry.text(),
            "include_subdirectories": self.include_subdirectories_switch.isChecked(),
        }
        self.is_running = True
        self.frame.setEnabled(False)
        self.create_masks_button.setEnabled(False)

        def _run():
            try:
                self.controller.create_masks(**arguments)
            except Exception:
                traceback.print_exc()
            finally:
                QTimer.singleShot(0, self, self._finish_generation)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _finish_generation(self):
        self.is_running = False
        self.frame.setEnabled(True)
        self.create_masks_button.setEnabled(True)

    def done(self, result):
        if not self.is_running:
            super().done(result)
