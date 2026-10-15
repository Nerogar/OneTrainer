import contextlib
from abc import ABC

from modules.ui.BaseConfigListView import BaseConfigListView
from modules.util.ui import ctk_components, dialogs

import customtkinter as ctk


class CtkConfigListView(BaseConfigListView, ABC):

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
        BaseConfigListView.__init__(self, ctk_components)

        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

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
        frame = ctk.CTkFrame(master, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        return frame

    def _create_element_list_frame(self, master):
        frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        frame.grid(row=1, column=0, sticky="nsew")
        if self.is_full_width:
            frame.grid_columnconfigure(0, weight=1)
        return frame

    def _wait_for_window(self, window):
        self.master.wait_window(window)

    def _remove_widget_from_layout(self, widget):
        widget.grid_remove()

    def _destroy_widget(self, widget):
        with contextlib.suppress(AttributeError):
            widget.destroy()

    def _destroy_frame(self, frame):
        frame.destroy()

    def _show_name_dialog(self, callback):
        dialogs.StringInputDialog(self.master, "name", "Name", callback)
