import faulthandler

from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from scalene import scalene_profiler


class ProfilingWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        self.title("Profiling")
        self.geometry("512x512")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        components.button(self, 0, 0, "Dump stack", self._dump_stack)
        self._profile_button = components.button(
            self, 1, 0, "Start Profiling", self._start_profiler,
            tooltip="Turns on/off Scalene profiling. Only works when OneTrainer is launched with Scalene!")

        # Bottom bar
        self._bottom_bar = ctk.CTkFrame(master=self, corner_radius=0)
        self._bottom_bar.grid(row=2, column=0, sticky="sew")
        self._message_label = components.label(self._bottom_bar, 0, 0, "Inactive")

        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.withdraw()
        self.after(200, lambda: set_window_icon(self))

    def _dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(f)
        self._message_label.configure(text='Stack dumped to stacks.txt')

    def _end_profiler(self):
        scalene_profiler.stop()

        self._message_label.configure(text='Inactive')
        self._profile_button.configure(text='Start Profiling')
        self._profile_button.configure(command=self._start_profiler)

    def _start_profiler(self):
        scalene_profiler.start()

        self._message_label.configure(text='Profiling active...')
        self._profile_button.configure(text='End Profiling')
        self._profile_button.configure(command=self._end_profiler)
