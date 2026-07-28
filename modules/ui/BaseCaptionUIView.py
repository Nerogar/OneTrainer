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
        self.components.button(frame, 0, 0, "打开", self.open_directory,
                          tooltip="打开新目录")
        self.components.button(frame, 0, 1, "生成遮罩", self.open_mask_window,
                          tooltip="打开自动生成遮罩对话框")
        self.components.button(frame, 0, 2, "生成标签", self.open_caption_window,
                          tooltip="打开自动生成标签对话框")

        if platform.system() == "Windows":
            self.components.button(frame, 0, 3, "在资源管理器中打开", self.open_in_explorer,
                              tooltip="在资源管理器中打开当前图像")

        self.components.switch(frame, 0, 4, ui_state, "include_subdirectories",
                          text="include subdirectories")

        frame.grid_columnconfigure(5, weight=1)

        self.components.button(frame, 0, 6, "帮助", controller.print_help,
                          tooltip=controller.help_text)

    def build_mask_buttons(self, right_frame):
        self.components.button(right_frame, 0, 0, "绘制", self.draw_mask_editing_mode,
                          tooltip="用画笔绘制遮罩")
        self.components.button(right_frame, 0, 1, "填充", self.fill_mask_editing_mode,
                          tooltip="用填充工具绘制遮罩")
