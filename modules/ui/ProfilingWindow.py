import faulthandler
import customtkinter as ctk

from modules.util.ui import components


class ProfilingWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("Profiling")
        self.geometry("1200x800")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        #self.grid_columnconfigure(1, weight=1)

        components.button(self, 0, 0, "Dump stack", self.dump_stack)
        components.button(self, 1, 0, "Do something", self.do_nothing)

    def do_nothing(self):
        pass

    def dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(f)
