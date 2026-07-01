
from modules.ui.BaseProfilingWindowView import BaseProfilingWindowView
from modules.ui.ProfilingWindowController import ProfilingWindowController
from modules.util.ui import ctk_components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkProfilingWindowView(BaseProfilingWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: ProfilingWindowController, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseProfilingWindowView.__init__(self, ctk_components)

        self._controller = controller

        self.title("Profiling")
        self.geometry("512x512")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Bottom bar
        self._bottom_bar = ctk.CTkFrame(master=self, corner_radius=0)
        self._bottom_bar.grid(row=2, column=0, sticky="sew")

        self.build_content(self, self._bottom_bar, controller)

        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.withdraw()
        self.after(200, lambda: set_window_icon(self))

    def set_message(self, text):
        self._message_label.configure(text=text)

    def set_profiling_active(self, active):
        if active:
            self._message_label.configure(text='Profiling active...')
            self._profile_button.configure(text='End Profiling')
            self._profile_button.configure(command=self._controller.end_profiler)
        else:
            self._message_label.configure(text='Inactive')
            self._profile_button.configure(text='Start Profiling')
            self._profile_button.configure(command=self._controller.start_profiler)
