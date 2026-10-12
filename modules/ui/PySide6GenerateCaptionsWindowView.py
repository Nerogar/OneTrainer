import threading

from modules.ui.BaseGenerateCaptionsWindowView import BaseGenerateCaptionsWindowView
from modules.ui.GenerateCaptionsWindowController import GenerateCaptionsWindowController
from modules.util.caption_ui_settings import load_caption_ui_settings, save_caption_ui_settings
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
    QTextEdit,
    QWidget,
)


class PySide6GenerateCaptionsWindowView(BaseGenerateCaptionsWindowView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: GenerateCaptionsWindowController, path, parent_include_subdirectories):
        QDialog.__init__(self, parent)
        self.controller = controller
        self._running = False
        self._cancel_requested = False

        settings = load_caption_ui_settings()

        self.setWindowTitle("Batch generate captions (LM Studio)")
        self.resize(540, 640)

        modes = ["Replace all captions", "Create if absent", "Add as new line"]

        lo = QGridLayout(self)
        lo.setContentsMargins(10, 10, 10, 10)
        lo.setColumnStretch(1, 1)
        row = 0

        self.url_edit = QLineEdit(settings["server_url"], self)
        self._add_row(lo, row, "Server URL", self.url_edit)
        row += 1

        self.path_edit = QLineEdit(path or "", self)
        self._add_row(lo, row, "Folder", self._path_row(self.path_edit))
        row += 1

        lo.addWidget(QLabel("System Prompt", self), row, 0, Qt.AlignLeft | Qt.AlignTop)
        self.system_edit = QTextEdit(self)
        self.system_edit.setPlainText(settings["system_prompt"])
        self.system_edit.setFixedHeight(90)
        lo.addWidget(self.system_edit, row, 1)
        row += 1

        lo.addWidget(QLabel("User Prompt", self), row, 0, Qt.AlignLeft | Qt.AlignTop)
        self.user_edit = QTextEdit(self)
        self.user_edit.setPlainText(settings["user_prompt"])
        self.user_edit.setFixedHeight(90)
        lo.addWidget(self.user_edit, row, 1)
        row += 1

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(modes)
        self.mode_combo.setCurrentText("Create if absent")
        self._add_row(lo, row, "Mode", self.mode_combo)
        row += 1

        self.include_subdirs_check = QCheckBox("Include subfolders", self)
        self.include_subdirs_check.setChecked(bool(parent_include_subdirectories))
        lo.addWidget(self.include_subdirs_check, row, 1)
        row += 1

        lo.addWidget(QLabel("Log", self), row, 0, Qt.AlignLeft | Qt.AlignTop)
        self.status_box = QTextEdit(self)
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(120)
        lo.addWidget(self.status_box, row, 1)
        row += 1

        self.progress_label = QLabel("Progress: 0/0", self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        lo.addWidget(self.progress_label, row, 0)
        lo.addWidget(self.progress_bar, row, 1)
        row += 1

        self.create_button = QPushButton("Create Captions", self)
        self.create_button.clicked.connect(self._on_create)
        lo.addWidget(self.create_button, row, 0)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel)
        lo.addWidget(self.cancel_button, row, 1)
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

    def _log(self, text):
        QTimer.singleShot(0, self, lambda: self.status_box.append(text))

    def _on_create(self):
        if self._running:
            return

        server_url = self.url_edit.text().strip()
        system_prompt = self.system_edit.toPlainText()
        user_prompt = self.user_edit.toPlainText()
        path = self.path_edit.text()
        mode_str = self.mode_combo.currentText()
        include_subdirectories = self.include_subdirs_check.isChecked()

        save_caption_ui_settings(server_url, system_prompt, user_prompt)

        self._running = True
        self._cancel_requested = False
        self.create_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_box.clear()
        self._log(f"Connecting to {server_url} ...")

        def error_callback(filename):
            self._log(f"Error captioning {filename}")

        def worker():
            try:
                self.controller.create_captions_lmstudio(
                    server_url=server_url,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    path=path,
                    mode_str=mode_str,
                    include_subdirectories=include_subdirectories,
                    progress_callback=self.set_progress,
                    error_callback=error_callback,
                    is_cancelled=lambda: self._cancel_requested,
                )
                self._log("Cancelled." if self._cancel_requested else "Done.")
            except Exception as e:
                message = str(e)
                self._log(f"Failed: {message}")
            finally:
                QTimer.singleShot(0, self, self._on_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_cancel(self):
        if self._running and not self._cancel_requested:
            self._cancel_requested = True
            self.cancel_button.setEnabled(False)
            self._log("Cancelling after the current image ...")

    def _on_done(self):
        self._running = False
        self.create_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
