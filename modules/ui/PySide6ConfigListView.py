import contextlib
from abc import ABC

from modules.ui.BaseConfigListView import BaseConfigListView
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QInputDialog, QWidget


class PySide6ConfigListView(BaseConfigListView, ABC):

    def __init__(
            self,
            master,
            controller,
            ui_state,
            from_external_file: bool,
            attr_name: str = "",
            enable_key: str = "enabled",
            config_dir: str = "",
            default_config_name: str = "",
            add_button_text: str = "",
            add_button_tooltip: str = "",
            is_full_width: bool = False,
            show_toggle_button: bool = False,
    ):
        BaseConfigListView.__init__(self, pyside6_components)

        master_lo = pyside6_components._layout(master)
        master_lo.setContentsMargins(
            pyside6_components.PAD, pyside6_components.PAD,
            pyside6_components.PAD, pyside6_components.PAD,
        )
        master_lo.setRowStretch(0, 0)
        master_lo.setRowStretch(1, 1)
        master_lo.setColumnStretch(0, 1)

        self.build(
            master, controller, ui_state, from_external_file,
            attr_name=attr_name,
            enable_key=enable_key,
            config_dir=config_dir,
            default_config_name=default_config_name,
            add_button_text=add_button_text,
            add_button_tooltip=add_button_tooltip,
            is_full_width=is_full_width,
            show_toggle_button=show_toggle_button,
        )

    def _create_top_frame(self, master):
        frame = QWidget(master)
        pyside6_components._layout(master).addWidget(frame, 0, 0)
        pyside6_components._layout(frame).setColumnStretch(4, 1)
        return frame

    def _create_element_list_frame(self, master):
        scroll, content = pyside6_components.scrollable_frame(master)
        pyside6_components._layout(master).addWidget(scroll, 1, 0)
        if self.is_full_width:
            pyside6_components._layout(content).setColumnStretch(0, 1)
        content._scroll_area = scroll
        return content

    def _wait_for_window(self, window):
        window.exec()

    def _remove_widget_from_layout(self, widget):
        widget.hide()

    def _destroy_widget(self, widget):
        with contextlib.suppress(RuntimeError, AttributeError):
            widget.hide()
            widget.deleteLater()

    def _destroy_frame(self, frame):
        with contextlib.suppress(RuntimeError, AttributeError):
            scroll = getattr(frame, '_scroll_area', None)
            if scroll is not None:
                lo = scroll.parent().layout() if scroll.parent() else None
                if lo is not None:
                    lo.removeWidget(scroll)
                scroll.hide()
                scroll.deleteLater()
            else:
                frame.hide()
                frame.deleteLater()

    def _update_toggle_button_text(self):
        if not self.show_toggle_button:
            return
        self._update_item_enabled_state()
        if self.toggle_button is not None:
            self.toggle_button.setText("Disable" if self._is_current_item_enabled else "Enable")

    def _show_name_dialog(self, callback):
        text, ok = QInputDialog.getText(self.master, "name", "Name")
        if ok and text:
            callback(text)
