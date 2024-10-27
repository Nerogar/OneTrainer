import faulthandler

from modules.util.ui import components

import customtkinter as ctk
from scalene import scalene_profiler


class ProfilingWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("分析")
        self.geometry("512x512")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        components.button(self, 0, 0, "查看堆栈", self._dump_stack)
        self._profile_button = components.button(
            self, 1, 0, "开始性能分析", self._start_profiler,
            tooltip="打开/关闭 Scalene 性能分析。只有当 OneTrainer 使用 Scalene 启动时才有效！")

        # Bottom bar
        self._bottom_bar = ctk.CTkFrame(master=self, corner_radius=0)
        self._bottom_bar.grid(row=2, column=0, sticky="sew")
        self._message_label = components.label(self._bottom_bar, 0, 0, "非活动状态")

        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.withdraw()

    def _dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(f)
        self._message_label.configure(text='堆栈已转储到 stacks.txt')

    def _end_profiler(self):
        scalene_profiler.stop()

        self._message_label.configure(text='非活动状态')
        self._profile_button.configure(text='开始性能分析')
        self._profile_button.configure(command=self._start_profiler)

    def _start_profiler(self):
        scalene_profiler.start()

        self._message_label.configure(text='性能分析已激活...')
        self._profile_button.configure(text='关闭性能分析')
        self._profile_button.configure(command=self._end_profiler)
