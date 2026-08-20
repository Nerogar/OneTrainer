import threading

from modules.ui.BaseGenerateMasksWindowView import BaseGenerateMasksWindowView
from modules.ui.GenerateMasksWindowController import GenerateMasksWindowController
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QWidget,
)


class PySide6GenerateMasksWindowView(BaseGenerateMasksWindowView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: GenerateMasksWindowController, path, parent_include_subdirectories):
        QDialog.__init__(self, parent)
        self.controller = controller
        self._running = False

        self.setWindowTitle("Batch generate masks")
        self.resize(400, 470)

        models = ["ClipSeg", "Rembg", "Rembg-Human", "Hex Color"]
        modes = ["Replace all masks", "Create if absent", "Add to existing",
                 "Subtract from existing", "Blend with existing"]

        lo = QGridLayout(self)
        lo.setContentsMargins(10, 10, 10, 10)
        lo.setColumnStretch(1, 1)
        row = 0

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(models)
        self._add_row(lo, row, "Model", self.model_combo)
        row += 1

        self.path_edit = QLineEdit(path or "", self)
        self._add_row(lo, row, "Folder", self._path_row(self.path_edit))
        row += 1

        self.prompt_edit = QLineEdit(self)
        self._add_row(lo, row, "Prompt", self.prompt_edit)
        row += 1

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(modes)
        self.mode_combo.setCurrentText("Create if absent")
        self._add_row(lo, row, "Mode", self.mode_combo)
        row += 1

        self.threshold_edit = QLineEdit("0.3", self)
        self._add_row(lo, row, "Threshold", self.threshold_edit)
        row += 1

        self.smooth_edit = QLineEdit("5", self)
        self._add_row(lo, row, "Smooth", self.smooth_edit)
        row += 1

        self.expand_edit = QLineEdit("10", self)
        self._add_row(lo, row, "Expand", self.expand_edit)
        row += 1

        self.alpha_edit = QLineEdit("1", self)
        self._add_row(lo, row, "Alpha", self.alpha_edit)
        row += 1

        self.include_subdirs_check = QCheckBox("Include subfolders", self)
        self.include_subdirs_check.setChecked(bool(parent_include_subdirectories))
        lo.addWidget(self.include_subdirs_check, row, 1)
        row += 1

        self.progress_label = QLabel("Progress: 0/0", self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        lo.addWidget(self.progress_label, row, 0)
        lo.addWidget(self.progress_bar, row, 1)
        row += 1

        self.create_button = QPushButton("Create Masks", self)
        self.create_button.clicked.connect(self._on_create)
        lo.addWidget(self.create_button, row, 0, 1, 2)
        row += 1

        lo.setRowStretch(row, 1)

    def _add_row(self, lo, row, label_text, widget):
        lo.addWidget(QLabel(label_text, self), row, 0, Qt.AlignLeft | Qt.AlignVCenter)
        lo.addWidget(widget, row, 1)

    def _path_row(self, edit):
        frame = QWidget(self)
        frame_lo = QGridLayout(frame)
        frame_lo.setContentsMargins(0, 0, 0, 0)
        frame_lo.setColumnStretch(0, 1)
        frame_lo.addWidget(edit, 0, 0)
        browse = QPushButton("...", frame)
        browse.setFixedWidth(40)
        browse.clicked.connect(self._browse)
        frame_lo.addWidget(browse, 0, 1)
        return frame

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(self, "Select folder", self.path_edit.text())
        if directory:
            self.path_edit.setText(directory)

    def set_progress(self, value, max_value):
        QTimer.singleShot(0, self, lambda: self._apply_progress(value, max_value))

    def _apply_progress(self, value, max_value):
        max_value = max(1, max_value)
        self.progress_bar.setValue(int(value / max_value * 100))
        self.progress_label.setText(f"Progress: {value}/{max_value}")

    def _on_create(self):
        if self._running:
            return
        self._running = True
        self.create_button.setEnabled(False)

        args = {
            "model_name": self.model_combo.currentText(),
            "path": self.path_edit.text(),
            "prompt": self.prompt_edit.text(),
            "mode_str": self.mode_combo.currentText(),
            "alpha_str": self.alpha_edit.text(),
            "threshold_str": self.threshold_edit.text(),
            "smooth_str": self.smooth_edit.text(),
            "expand_str": self.expand_edit.text(),
            "include_subdirectories": self.include_subdirs_check.isChecked(),
        }

        def worker():
            try:
                self.controller.create_masks(**args)
            except Exception as e:
                message = str(e)
                QTimer.singleShot(0, self, lambda: self.progress_label.setText(f"Error: {message}"))
            finally:
                QTimer.singleShot(0, self, self._on_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self):
        self._running = False
        self.create_button.setEnabled(True)
