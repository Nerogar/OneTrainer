from abc import abstractmethod


class BaseProfilingWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, bottom_bar, controller):
        self.components.button(frame, 0, 0, "Dump stack", controller.dump_stack)
        self._profile_button = self.components.button(
            frame, 1, 0, "Start Profiling", controller.start_profiler,
            tooltip="Turns on/off Scalene profiling. Only works when OneTrainer is launched with Scalene!")
        self._message_label = self.components.label(bottom_bar, 0, 0, "Inactive")

    @abstractmethod
    def set_message(self, text):
        pass

    @abstractmethod
    def set_profiling_active(self, active):
        pass
