import contextlib
from tkinter import TclError

from modules.ui.BaseOptimizerParamsWindowView import BaseOptimizerParamsWindowView
from modules.ui.CtkMuonAdamWindowView import CtkMuonAdamWindowView
from modules.ui.MuonAdamWindowController import MuonAdamWindowController
from modules.ui.OptimizerParamsWindowController import OptimizerParamsWindowController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkOptimizerParamsWindowView(BaseOptimizerParamsWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: OptimizerParamsWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseOptimizerParamsWindowView.__init__(self, ctk_components)

        self.controller = controller
        self.ui_state = ui_state
        self.optimizer_ui_state = ui_state.get_var("optimizer")
        self.muon_adam_button = None
        self.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.title("Optimizer Settings")
        self.geometry("800x500")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, minsize=50)
        self.frame.grid_columnconfigure(3, weight=0)
        self.frame.grid_columnconfigure(4, weight=1)

        self.components.button(self, 1, 0, "ok", command=self.on_window_close)
        self.build_content(self.frame, controller, ui_state, self.optimizer_ui_state,
                           self.on_optimizer_change, self._load_defaults)
        self._rebuild_dynamic_ui()

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def _rebuild_dynamic_ui(self):
        with contextlib.suppress(TclError):
            for widget in self.frame.winfo_children():
                grid_info = widget.grid_info()
                if int(grid_info["row"]) >= 1:
                    widget.destroy()

        if not self.winfo_exists():
            return

        self.build_dynamic_content(self.frame, self.controller, self.optimizer_ui_state,
                                   self.update_user_pref, self.open_muon_adam_window)
        self.toggle_muon_adam_button()

    def update_user_pref(self, *args):
        self.controller.on_close()
        self.toggle_muon_adam_button()

    def on_optimizer_change(self, *args):
        self.controller.restore_optimizer_config(self.ui_state)
        self._rebuild_dynamic_ui()

    def _load_defaults(self, *args):
        self.controller.load_defaults(self.ui_state)

    def on_window_close(self):
        self.destroy()

    def toggle_muon_adam_button(self):
        if self.muon_adam_button and self.muon_adam_button.winfo_exists():
            muon_with_adam = self.optimizer_ui_state.get_var("MuonWithAuxAdam").get()
            self.muon_adam_button.configure(state="normal" if muon_with_adam else "disabled")

    def open_muon_adam_window(self):
        adam_config, current_optimizer = self.controller.prepare_muon_adam_config()
        temp_adam_ui_state = CtkUIState(self, adam_config)
        window = CtkMuonAdamWindowView(self, MuonAdamWindowController(self.controller.config, current_optimizer), temp_adam_ui_state)
        self.wait_window(window)
        self.controller.save_muon_adam_config(adam_config)
