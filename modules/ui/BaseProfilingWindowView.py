from abc import abstractmethod


class BaseProfilingWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, bottom_bar, controller):
        self.components.button(frame, 0, 0, "Dump stack", controller.dump_stack)
        self._profile_button = self.components.button(
            frame, 1, 0, "开始分析", controller.start_profiler,
            tooltip="开关Scalene性能分析，仅在使用Scalene启动时有效")
        self._message_label = self.components.label(bottom_bar, 0, 0, "未激活")

    @abstractmethod
    def set_message(self, text):
        pass

    @abstractmethod
    def set_profiling_active(self, active):
        pass
