
from modules.ui.BaseConfigListView import BaseConfigListView


class BaseSchedulerParamsWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, master, controller, ui_state):
        if controller.is_custom_scheduler():
            self.components.label(master, 0, 0, "类名",
                                  tooltip="自定义调度器类，格式：<模块>.<类名>")
            self.components.entry(master, 0, 1, ui_state, "custom_learning_rate_scheduler")


class BaseKvParamsView(BaseConfigListView):
    def __init__(self, components):
        self.components = components

    def open_element_window(self, i, ui_state):
        pass
