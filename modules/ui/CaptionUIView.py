import platform

from modules.ui.CaptionUIController import CaptionUIController
from modules.ui.GenerateCaptionsWindowView import GenerateCaptionsWindowView
from modules.ui.GenerateMasksWindowView import GenerateMasksWindowView
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)


class CaptionUIView(QDialog):
    def __init__(self, parent, controller: CaptionUIController):
        QDialog.__init__(self, parent)
        self.components = pyside6_components

        self.controller = controller
        controller.view = self
        self.config_ui_state = PySide6UIState(controller.config_ui_data)
        self.prompt_component = None
        self._generation_window = None
        self._closed = False

        self.setWindowTitle("OneTrainer")
        self.setWindowIcon(QIcon("resources/icons/icon.png"))
        self.setFixedSize(1280, 980)
        self.finished.connect(self.deleteLater)
        self.destroyed.connect(controller._release_models)

        outer = QGridLayout(self)
        pad = pyside6_components.PAD
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(pyside6_components.PAD)
        outer.setRowStretch(1, 1)
        outer.setColumnStretch(1, 1)

        top_frame = QWidget(self)
        self.build_top_bar(top_frame, controller, self.config_ui_state)
        pyside6_components._layout(top_frame).setColumnStretch(5, 1)
        outer.addWidget(top_frame, 0, 0, 1, 2)

        self.file_list = QListWidget(self)
        self.file_list.setFixedWidth(300)
        self.file_list.setFocusPolicy(Qt.NoFocus)
        self.file_list.itemClicked.connect(lambda item: controller.switch_image(self.file_list.row(item)))
        outer.addWidget(self.file_list, 1, 0)

        right_frame = QWidget(self)
        right_layout = pyside6_components._layout(right_frame)
        right_layout.setColumnStretch(4, 1)
        right_layout.setRowStretch(1, 1)
        outer.addWidget(right_frame, 1, 1)
        self.build_mask_buttons(right_frame)

        self.enable_mask_editing = QCheckBox("Enable Mask Editing", right_frame)
        right_layout.addWidget(self.enable_mask_editing, 0, 2)
        self.mask_editing_alpha = QLineEdit("1.0", right_frame)
        self.mask_editing_alpha.setFixedWidth(40)
        self.mask_editing_alpha.setPlaceholderText("1.0")
        right_layout.addWidget(self.mask_editing_alpha, 0, 3)
        pyside6_components.label(right_frame, 0, 4, "Brush Alpha")

        image_frame = QWidget(right_frame)
        image_frame.setFixedSize(controller.image_size, controller.image_size)
        image_layout = QGridLayout(image_frame)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel(image_frame)
        self.image_label.setFixedSize(controller.image_size, controller.image_size)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        # Only the image-sized label receives edits, keeping coordinates image-relative.
        image_layout.addWidget(self.image_label, 0, 0, Qt.AlignCenter)
        right_layout.addWidget(image_frame, 1, 0, 1, 5, Qt.AlignCenter)

        self.prompt_component = QLineEdit(right_frame)
        right_layout.addWidget(self.prompt_component, 2, 0, 1, 5)
        self.prompt_component.installEventFilter(self)
        self.mask_editing_alpha.installEventFilter(self)

        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

        controller.load_directory()

    def refresh_file_list(self):
        self.file_list.clear()
        self.file_list.addItems(self.controller.image_rel_paths)

    def focus_prompt(self):
        self.prompt_component.setFocus()

    def on_image_switched(self, old_index, new_index, prompt):
        self.file_list.setCurrentRow(new_index)
        self.refresh_image()
        self.prompt_component.setText(prompt)

    def on_image_cleared(self):
        image = Image.new("RGB", (self.image_label.width(), self.image_label.height()), (0, 0, 0))
        self.image_label.setPixmap(QPixmap.fromImage(ImageQt(image)))

    def refresh_image(self):
        pil_image, size = self.controller.get_display_image()
        self.image_label.setFixedSize(*size)
        self.image_label.setPixmap(QPixmap.fromImage(ImageQt(pil_image.convert("RGBA"))))

    def eventFilter(self, watched, event):
        if watched in (self.prompt_component, self.mask_editing_alpha) and event.type() == QEvent.Type.KeyPress:
            commands = {
                (Qt.Key_Up, Qt.NoModifier): self.controller.previous_image,
                (Qt.Key_Down, Qt.NoModifier): self.controller.next_image,
                (Qt.Key_Return, Qt.NoModifier): self.save,
                (Qt.Key_Enter, Qt.NoModifier): self.save,
                (Qt.Key_M, Qt.ControlModifier): self.toggle_mask,
                (Qt.Key_D, Qt.ControlModifier): self.draw_mask_editing_mode,
                (Qt.Key_F, Qt.ControlModifier): self.fill_mask_editing_mode,
            }
            command = commands.get((event.key(), event.modifiers()))
            if command is not None:
                command()
                return True
        elif watched is self.image_label:
            if event.type() in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress):
                self.edit_mask(event)
                return True
            if event.type() == QEvent.Type.Wheel:
                self.controller.update_mask_draw_radius(int(event.angleDelta().y() / 120))
                return True
        return super().eventFilter(watched, event)

    def edit_mask(self, event):
        if not self.enable_mask_editing.isChecked():
            return

        buttons = event.buttons() | event.button()
        is_left = bool(buttons & Qt.LeftButton)
        is_right = not is_left and bool(buttons & Qt.RightButton)
        try:
            alpha = float(self.mask_editing_alpha.text())
        except ValueError:
            alpha = 1.0

        position = event.position()
        self.controller.handle_edit_mask(position.x(), position.y(), is_left, is_right, alpha)

    def save(self):
        self.controller.save(self.prompt_component.text())

    def draw_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode("draw")

    def fill_mask_editing_mode(self, *args):
        self.controller.set_mask_editing_mode("fill")

    def toggle_mask(self):
        self.controller.toggle_mask()
        self.refresh_image()

    def open_directory(self):
        new_dir = QFileDialog.getExistingDirectory(self)
        if new_dir:
            self.controller.dir = new_dir
            self.controller.load_directory(self.controller.config_ui_data["include_subdirectories"])

    def _open_generation_window(self, window):
        self._generation_window = window
        try:
            window.exec()
        finally:
            self._generation_window = None
        if not self._closed:
            self.controller.switch_image(self.controller.current_image_index)

    def open_mask_window(self):
        self._open_generation_window(self.controller.open_mask_window(self, GenerateMasksWindowView))

    def open_caption_window(self):
        self._open_generation_window(self.controller.open_caption_window(self, GenerateCaptionsWindowView))

    def open_in_explorer(self):
        self.controller.open_in_explorer()

    def done(self, result):
        if self._generation_window is not None and self._generation_window.is_running:
            return
        if self._closed:
            return
        self._closed = True
        if self._generation_window is not None:
            self._generation_window.reject()
        self.controller._release_models()
        super().done(result)

    def destroy(self):
        self.close()

    def build_top_bar(self, frame, controller, ui_state):
        self.components.button(frame, 0, 0, "Open", self.open_directory,
                          tooltip="open a new directory")
        self.components.button(frame, 0, 1, "Generate Masks", self.open_mask_window,
                          tooltip="open a dialog to automatically generate masks")
        self.components.button(frame, 0, 2, "Generate Captions", self.open_caption_window,
                          tooltip="open a dialog to automatically generate captions")

        if platform.system() == "Windows":
            self.components.button(frame, 0, 3, "Open in Explorer", self.open_in_explorer,
                              tooltip="open the current image in Explorer")

        self.components.switch(frame, 0, 4, ui_state, "include_subdirectories",
                          text="include subdirectories")

        self.components.button(frame, 0, 6, "Help", controller.print_help,
                          tooltip=controller.help_text)

    def build_mask_buttons(self, right_frame):
        self.components.button(right_frame, 0, 0, "Draw", self.draw_mask_editing_mode,
                          tooltip="draw a mask using a brush")
        self.components.button(right_frame, 0, 1, "Fill", self.fill_mask_editing_mode,
                          tooltip="draw a mask using a fill tool")
