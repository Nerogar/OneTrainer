import customtkinter as ctk

from modules.util.ui import components
from modules.util.ui.UIState import UIState


class ConceptWindow(ctk.CTkToplevel):
    def __init__(self, parent, concept, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.ui_state = UIState(self, concept)

        self.title("Concept")
        self.geometry("380x300")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_columnconfigure(1, weight=1)

        # name
        components.label(self, 0, 0, "Name")
        components.entry(self, 0, 1, self.ui_state, "name")

        # path
        components.label(self, 1, 0, "Path")
        components.dir_entry(self, 1, 1, self.ui_state, "path")

        # flip
        components.label(self, 2, 0, "Random Flip")
        components.switch(self, 2, 1, self.ui_state, "random_flip")

        components.button(self, 3, 0, "ok", self.__ok)

    def __ok(self):
        self.destroy()
