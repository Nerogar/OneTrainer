import platform
from abc import ABC, abstractmethod


class BaseCaptionUIView(ABC):
    def __init__(self, components):
        self.components = components

    @abstractmethod
    def open_directory(self): pass

    @abstractmethod
    def open_mask_window(self): pass

    @abstractmethod
    def open_caption_window(self): pass

    @abstractmethod
    def open_in_explorer(self): pass

    @abstractmethod
    def draw_mask_editing_mode(self, *args): pass

    @abstractmethod
    def fill_mask_editing_mode(self, *args): pass

    def build_top_bar(self, frame, controller, ui_state):
        self.components.button(frame, 0, 0, "Open", self.open_directory,
                          tooltip="open a new directory")
        self.components.button(frame, 0, 1, "Generate Masks", self.open_mask_window,
                          tooltip="open a dialog to automatically generate masks")
        self.components.button(frame, 0, 2, "Generate Captions", self.open_caption_window,
                          tooltip="open a dialog to automatically generate captions")

        if platform.system() == "Windows":
            self.components.button(frame, 0, 3, "Open in Explorer", self.open_in_explorer,
                              tooltip="open the current image in Explorer")

        self.components.switch(frame, 0, 4, ui_state, "include_subdirectories",
                          text="include subdirectories")

        frame.grid_columnconfigure(5, weight=1)

        self.components.button(frame, 0, 6, "Help", controller.print_help,
                          tooltip=controller.help_text)

    def build_mask_buttons(self, right_frame):
        self.components.button(right_frame, 0, 0, "Draw", self.draw_mask_editing_mode,
                          tooltip="draw a mask using a brush")
        self.components.button(right_frame, 0, 1, "Fill", self.fill_mask_editing_mode,
                          tooltip="draw a mask using a fill tool")
