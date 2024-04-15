import cProfile
import faulthandler
import customtkinter as ctk

from modules.util.ui import components


class ProfilingWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
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
            self, 1, 0, "Start CPU Profiling", self._start_cpu_profiler)

        # Bottom bar
        self._bottom_bar = ctk.CTkFrame(master=self, corner_radius=0)
        self._bottom_bar.grid(row=2, column=0, sticky="sew")
        self._message_label = components.label(self._bottom_bar, 0, 0, "Inactive")
        self._bottom_bar

    def _dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(f)
        self._message_label.configure(text='Stack dumped to stacks.txt')

    def _end_cpu_profiler(self):
        self._active_profile.create_stats()
        self._active_profile.dump_stats('cpu_profile.txt')

        self._message_label.configure(text='Profile dumped to cpu_profile.txt')
        self._profile_button.configure(text='Start CPU Profiling')
        self._profile_button.configure(command=self._start_cpu_profiler)

    def _start_cpu_profiler(self):
        self._active_profile = cProfile.Profile()
        self._active_profile.enable()

        self._message_label.configure(text='Profiling active...')
        self._profile_button.configure(text='End CPU Profiling')
        self._profile_button.configure(command=self._end_cpu_profiler)
