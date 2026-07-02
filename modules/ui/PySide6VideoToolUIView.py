from modules.ui.BaseVideoToolUIView import BaseVideoToolUIView
from modules.ui.VideoToolUIController import VideoToolUIController
from modules.util.image_util import load_image
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QWidget,
)

_PAD = pyside6_components.PAD


class PySide6VideoToolUIView(BaseVideoToolUIView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: VideoToolUIController):
        QDialog.__init__(self, parent)
        BaseVideoToolUIView.__init__(self, pyside6_components)

        self.controller = controller
        self._status_box: QTextEdit | None = None
        self._preview_label: QLabel | None = None
        self._preview_caption_label: QLabel | None = None

        ui_state = PySide6UIState(controller.args)

        self.setWindowTitle("Video Tools")
        self.resize(700, 750)

        outer = QGridLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.setRowStretch(0, 1)
        outer.setRowStretch(1, 0)

        tabs = QTabWidget(self)
        outer.addWidget(tabs, 0, 0)

        for name, build_fn in [
            ("extract clips", self.build_clip_extract_tab),
            ("extract images", self.build_image_extract_tab),
            ("download", self.build_video_download_tab),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            frame = QWidget()
            scroll.setWidget(frame)
            lo = pyside6_components._layout(frame)
            lo.setContentsMargins(_PAD, _PAD, _PAD, _PAD)
            lo.setColumnMinimumWidth(0, 120)
            lo.setColumnStretch(3, 1)
            build_fn(frame, controller, ui_state)
            lo.setRowStretch(lo.rowCount(), 1)
            tabs.addTab(scroll, name)

        outer.addWidget(self._build_status_bar(), 1, 0)

    def _build_status_bar(self):
        frame = QWidget(self)
        lo = QGridLayout(frame)
        lo.setColumnMinimumWidth(0, 160)
        lo.setColumnStretch(2, 1)

        self._preview_label = QLabel(frame)
        self._preview_label.setFixedSize(150, 150)
        preview = load_image("resources/icons/icon.png", 'RGB')
        preview.thumbnail((150, 150))
        self._preview_label.setPixmap(
            QPixmap.fromImage(ImageQt(preview.convert("RGBA"))).scaled(
                150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self._preview_caption_label = QLabel("Preview image", frame)
        self._preview_caption_label.setWordWrap(True)

        preview_col = QWidget(frame)
        preview_lo = QGridLayout(preview_col)
        preview_lo.setContentsMargins(0, 0, 0, 0)
        preview_lo.addWidget(self._preview_label, 0, 0, Qt.AlignTop)
        preview_lo.addWidget(self._preview_caption_label, 1, 0, Qt.AlignTop)
        lo.addWidget(preview_col, 0, 0, Qt.AlignTop | Qt.AlignLeft)

        self._status_box = QTextEdit(frame)
        self._status_box.setReadOnly(True)
        self._status_box.setFixedHeight(160)
        self._status_box.setMinimumWidth(300)
        self._status_box.setPlainText("Current status")
        lo.addWidget(self._status_box, 0, 1, Qt.AlignTop)

        return frame

    # --- abstract method implementations ---

    def _create_textbox(self, master, row, col, width, height, ui_state, var_name):
        var = ui_state.get_var(var_name)
        widget = QTextEdit(master)
        widget.setFixedHeight(height)
        widget.setMinimumWidth(width)
        widget.setPlainText(var.get())
        pyside6_components._add(
            pyside6_components._layout(master), widget, row, col, sticky="w", rowspan=2
        )
        widget.textChanged.connect(lambda: var.set(widget.toPlainText()))
        return widget

    def update_status(self, status_text: str):
        self._status_box.append(status_text)

    def clear_status(self):
        self._status_box.clear()

    def update_preview(self, preview_image, label_text: str):
        pixmap = QPixmap.fromImage(ImageQt(preview_image.convert("RGBA")))
        self._preview_label.setPixmap(
            pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self._preview_caption_label.setText(label_text)
