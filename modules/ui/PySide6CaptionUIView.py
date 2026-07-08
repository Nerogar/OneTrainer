import platform
import threading

from modules.ui.BaseCaptionUIView import BaseCaptionUIView
from modules.ui.PySide6GenerateCaptionsWindowView import PySide6GenerateCaptionsWindowView
from modules.ui.PySide6GenerateMasksWindowView import PySide6GenerateMasksWindowView
from modules.util.caption_ui_settings import load_caption_ui_settings
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _MaskCanvas(QLabel):
    """Displays the current image/mask and forwards mouse edits to the controller."""

    def __init__(self, controller, get_alpha, is_editing_enabled, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._get_alpha = get_alpha
        self._is_editing_enabled = is_editing_enabled
        self._pixmap_size = (0, 0)
        self._cursor_pos = None
        self._cursor_inside = False
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(controller.image_size, controller.image_size)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def set_display_pixmap(self, pixmap: QPixmap):
        self._pixmap_size = (pixmap.width(), pixmap.height())
        self.setPixmap(pixmap)

    def _brush_display_radius(self) -> float:
        # draw_mask sizes the brush as radius * max(mask dimensions); the mask is
        # uniformly scaled to the display pixmap, so max(pixmap) gives display px
        pw, ph = self._pixmap_size
        return self._controller.mask_draw_radius * max(pw, ph)

    def _map_to_image(self, pos):
        pw, ph = self._pixmap_size
        if pw == 0 or ph == 0:
            return None
        offset_x = (self.width() - pw) / 2
        offset_y = (self.height() - ph) / 2
        x = pos.x() - offset_x
        y = pos.y() - offset_y
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return None
        return x, y

    def enterEvent(self, event):
        self._cursor_inside = True
        self.update()

    def leaveEvent(self, event):
        self._cursor_inside = False
        self.update()

    def mousePressEvent(self, event):
        self._cursor_pos = event.position()
        if self._is_editing_enabled():
            mapped = self._map_to_image(event.position())
            if mapped is not None:
                self._controller.set_mask_draw_origin(mapped[0], mapped[1])
        self._handle(event)

    def mouseMoveEvent(self, event):
        self._cursor_pos = event.position()
        self._handle(event)
        if self._is_editing_enabled():
            self.update()

    def _handle(self, event):
        if not self._is_editing_enabled():
            return
        mapped = self._map_to_image(event.position())
        if mapped is None:
            return
        buttons = event.buttons()
        is_left = bool(buttons & Qt.LeftButton)
        is_right = bool(buttons & Qt.RightButton)
        if not is_left and not is_right:
            return
        self._controller.handle_edit_mask(mapped[0], mapped[1], is_left, is_right, self._get_alpha())

    def wheelEvent(self, event):
        if not self._is_editing_enabled():
            return
        delta = 1 if event.angleDelta().y() > 0 else -1
        self._controller.update_mask_draw_radius(delta)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not (self._is_editing_enabled() and self._cursor_inside and self._cursor_pos):
            return
        if self._pixmap_size == (0, 0):
            return

        radius = self._brush_display_radius()
        cx, cy = self._cursor_pos.x(), self._cursor_pos.y()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        # draw a dark ring then a dashed light ring so it stays visible on any image
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1.5))
        painter.drawEllipse(self._cursor_pos, radius, radius)
        pen = QPen(QColor(255, 255, 255, 220), 1.0)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(self._cursor_pos, radius, radius)
        # a small center dot marks the exact brush center
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.0))
        painter.drawPoint(int(cx), int(cy))
        painter.end()


class PySide6CaptionUIView(BaseCaptionUIView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller):
        QDialog.__init__(self, parent)
        BaseCaptionUIView.__init__(self, pyside6_components)

        self.controller = controller
        controller.view = self

        self.setWindowTitle("OneTrainer - Dataset Tool")
        self.resize(1280, 980)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_top_bar())

        body = QHBoxLayout()
        body.setSpacing(8)
        root.addLayout(body, 1)

        self.file_list = QListWidget(self)
        self.file_list.setFixedWidth(300)
        self.file_list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self.file_list)

        body.addWidget(self._build_content_column(), 1)

        self._install_shortcuts()
        self.controller.load_directory()

    # ---- layout builders ------------------------------------------------

    def _build_top_bar(self) -> QWidget:
        bar = QWidget(self)
        lo = QHBoxLayout(bar)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(6)

        def add(text, slot, tooltip):
            b = QPushButton(text, bar)
            b.setToolTip(tooltip)
            b.clicked.connect(slot)
            lo.addWidget(b)
            return b

        add("Open", self.open_directory, "open a new directory")
        add("Generate Masks", self.open_mask_window, "batch-generate masks")
        add("Generate Captions", self.open_caption_window, "batch-generate captions with LM Studio")
        self.caption_image_button = add(
            "Caption Image", self.caption_current_image,
            "caption the current image with LM Studio")
        if platform.system() == "Windows":
            add("Open in Explorer", self.open_in_explorer, "open the current image in Explorer")

        self.include_subdirs_check = QCheckBox("include subdirectories", bar)
        self.include_subdirs_check.setChecked(
            bool(self.controller.config_ui_data.get("include_subdirectories", False)))
        self.include_subdirs_check.toggled.connect(self._on_include_subdirs_toggled)
        lo.addWidget(self.include_subdirs_check)

        lo.addStretch(1)

        help_button = QPushButton("Help", bar)
        help_button.setToolTip(self.controller.help_text)
        help_button.clicked.connect(self.controller.print_help)
        lo.addWidget(help_button)
        return bar

    def _build_content_column(self) -> QWidget:
        column = QWidget(self)
        lo = QVBoxLayout(column)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(8)

        # mask controls row
        controls = QWidget(column)
        controls_lo = QHBoxLayout(controls)
        controls_lo.setContentsMargins(0, 0, 0, 0)
        controls_lo.setSpacing(6)

        draw_button = QPushButton("Draw", controls)
        draw_button.setToolTip("draw a mask using a brush")
        draw_button.clicked.connect(self.draw_mask_editing_mode)
        controls_lo.addWidget(draw_button)

        fill_button = QPushButton("Fill", controls)
        fill_button.setToolTip("draw a mask using a fill tool")
        fill_button.clicked.connect(self.fill_mask_editing_mode)
        controls_lo.addWidget(fill_button)

        self.enable_mask_editing_check = QCheckBox("Enable Mask Editing", controls)
        controls_lo.addWidget(self.enable_mask_editing_check)

        controls_lo.addSpacing(20)
        controls_lo.addWidget(QLabel("Brush Alpha", controls))
        self.mask_alpha_edit = QLineEdit("1.0", controls)
        self.mask_alpha_edit.setFixedWidth(50)
        controls_lo.addWidget(self.mask_alpha_edit)
        controls_lo.addStretch(1)
        lo.addWidget(controls)

        # image canvas
        self.canvas = _MaskCanvas(
            self.controller,
            self._get_alpha,
            self.enable_mask_editing_check.isChecked,
            column,
        )
        lo.addWidget(self.canvas, 1, Qt.AlignCenter)

        # prompt entry
        self.prompt_edit = QLineEdit(column)
        self.prompt_edit.returnPressed.connect(self._save)
        lo.addWidget(self.prompt_edit)
        return column

    def _install_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Up), self, self.controller.previous_image)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.controller.next_image)
        QShortcut(QKeySequence("Ctrl+M"), self, self._toggle_mask)
        QShortcut(QKeySequence("Ctrl+D"), self, self.draw_mask_editing_mode)
        QShortcut(QKeySequence("Ctrl+F"), self, self.fill_mask_editing_mode)

    # ---- helpers --------------------------------------------------------

    def _get_alpha(self) -> float:
        try:
            return float(self.mask_alpha_edit.text())
        except ValueError:
            return 1.0

    def _on_row_changed(self, index):
        if index >= 0 and index != self.controller.current_image_index:
            self.controller.switch_image(index)

    def _on_include_subdirs_toggled(self, checked):
        self.controller.config_ui_data["include_subdirectories"] = checked

    def _save(self):
        self.controller.save(self.prompt_edit.text())

    def _toggle_mask(self):
        self.controller.toggle_mask()
        self.refresh_image()

    # ---- view callbacks invoked by the controller -----------------------

    def refresh_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems(self.controller.image_rel_paths)
        self.file_list.blockSignals(False)

    def focus_prompt(self):
        self.prompt_edit.setFocus()

    def on_image_switched(self, old_index, new_index, prompt):
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(new_index)
        self.file_list.blockSignals(False)
        self.refresh_image()
        self.prompt_edit.setText(prompt)

    def on_image_cleared(self):
        blank = Image.new("RGB", (self.controller.image_size, self.controller.image_size), (0, 0, 0))
        self.canvas.set_display_pixmap(self._to_pixmap(blank))

    def refresh_image(self):
        pil_image, _size = self.controller.get_display_image()
        self.canvas.set_display_pixmap(self._to_pixmap(pil_image))

    @staticmethod
    def _to_pixmap(pil_image) -> QPixmap:
        return QPixmap.fromImage(ImageQt(pil_image.convert("RGBA")))

    # ---- abstract method implementations --------------------------------

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select folder", self.controller.dir or "")
        if directory:
            self.controller.dir = directory
            self.controller.load_directory(
                include_subdirectories=self.controller.config_ui_data["include_subdirectories"])

    def open_mask_window(self):
        window = self.controller.open_mask_window(self, PySide6GenerateMasksWindowView)
        window.exec()
        self.controller.switch_image(self.controller.current_image_index)

    def open_caption_window(self):
        window = self.controller.open_caption_window(self, PySide6GenerateCaptionsWindowView)
        window.exec()
        self.controller.switch_image(self.controller.current_image_index)

    def open_in_explorer(self):
        self.controller.open_in_explorer()

    def draw_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode("draw")

    def fill_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode("fill")

    # ---- single-image LM Studio captioning ------------------------------

    def caption_current_image(self):
        settings = load_caption_ui_settings()
        self.caption_image_button.setEnabled(False)

        def worker():
            try:
                caption = self.controller.caption_current_image(
                    settings["server_url"], settings["system_prompt"], settings["user_prompt"])
            except Exception as e:
                message = str(e)
                QTimer.singleShot(0, self, lambda: self._caption_failed(message))
                return
            QTimer.singleShot(0, self, lambda: self._caption_done(caption))

        threading.Thread(target=worker, daemon=True).start()

    def _caption_done(self, caption):
        self.caption_image_button.setEnabled(True)
        if caption:
            self.prompt_edit.setText(caption)

    def _caption_failed(self, message):
        self.caption_image_button.setEnabled(True)
        QMessageBox.warning(self, "Caption failed", message)

    # ---- lifecycle ------------------------------------------------------

    def closeEvent(self, event):
        self.controller._release_models()
        super().closeEvent(event)
