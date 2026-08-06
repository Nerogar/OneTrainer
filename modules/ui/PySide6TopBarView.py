from collections.abc import Callable

from modules.ui.BaseTopBarView import BaseTopBarView
from modules.ui.TopBarController import TopBarController
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import pyside6_components, theme

from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton, QWidget


class PySide6TopBarView(BaseTopBarView, QWidget):

    def __init__(
            self,
            master,
            controller: TopBarController,
            ui_state,
            change_model_type_callback: Callable[[ModelType], None],
            change_training_method_callback: Callable[[TrainingMethod], None],
            load_preset_callback: Callable[[], None],
    ):
        QWidget.__init__(self, master)
        BaseTopBarView.__init__(self, pyside6_components)

        self.frame = QWidget(self)
        pyside6_components._layout(self).addWidget(self.frame, 0, 0)
        pyside6_components._layout(self.frame).setContentsMargins(
            pyside6_components.PAD, pyside6_components.PAD,
            pyside6_components.PAD, pyside6_components.PAD,
        )

        self.build(self.frame, master, controller, ui_state,
                   change_model_type_callback, change_training_method_callback, load_preset_callback)

        # Theme switch. Lives in the PySide6 view, not the shared BaseTopBarView: the ctk top bar
        # has no color scheme to switch. A text glyph follows the palette's ButtonText.
        self.theme_button = pyside6_components.button(
            self.frame, 0, 8, "◐", self.__toggle_theme,
            tooltip="Switch between light and dark mode", sticky="v",
        )
        # Square and icon-sized: the grid stretches a button to its column otherwise. The side
        # comes from a text-less button so the glyph's own metrics don't set it.
        side = QPushButton().sizeHint().height()
        self.theme_button.setFixedSize(side, side)

    def __toggle_theme(self):
        theme.apply_theme(QApplication.instance(), dark=not theme.is_dark_theme)

    def _setup_frame_column_weight(self):
        pyside6_components._layout(self.frame).setColumnStretch(5, 1)

    def _forget_dropdown(self, widget):
        lo = pyside6_components._layout(self.frame)
        lo.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def _show_save_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", initial_dir, "JSON (*.json)")
        if path:
            # the native dialog doesn't reliably append the filter's extension on every platform
            if not path.endswith(".json"):
                path += ".json"
            callback(path)

    def _show_open_dialog(self, initial_dir: str, callback):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", initial_dir, "JSON (*.json)")
        if path:
            callback(path)
