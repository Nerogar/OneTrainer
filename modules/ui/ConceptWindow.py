import customtkinter as ctk

from modules.util.ui import components
from modules.util.ui.UIState import UIState


class ConceptWindow(ctk.CTkToplevel):
    def __init__(self, parent, concept, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.ui_state = UIState(self, concept)

        self.title("Concept")
        self.geometry("600x300")
        self.resizable(False, False)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # name
        components.label(self, 0, 0, "Name")
        components.entry(self, 0, 1, self.ui_state, "name")

        # path
        components.label(self, 1, 0, "Path")
        components.dir_entry(self, 1, 1, self.ui_state, "path")

        # flip
        components.label(self, 2, 0, "Random Flip")
        components.switch(self, 2, 1, self.ui_state, "enable_random_flip")

        # crop jitter
        components.label(self, 3, 0, "Crop Jitter")
        components.switch(self, 3, 1, self.ui_state, "enable_crop_jitter")

        # prompt mode
        components.label(self, 4, 0, "Prompt Source")
        prompt_path_entry = components.file_entry(self, 4, 2, self.ui_state, "prompt_path")

        def set_prompt_path_entry_enabled(option: str):
            if option == 'concept':
                for child in prompt_path_entry.children.values():
                    child.configure(state="normal")
            else:
                for child in prompt_path_entry.children.values():
                    child.configure(state="disabled")

        components.options_kv(self, 4, 1, [
            ("From text file per sample", 'sample'),
            ("From single text file", 'concept'),
            ("From image file name", 'filename'),
        ], self.ui_state, "prompt_source", command=set_prompt_path_entry_enabled)
        set_prompt_path_entry_enabled(concept["prompt_source"])

        components.button(self, 5, 0, "ok", self.__ok)

    def __ok(self):
        self.destroy()
