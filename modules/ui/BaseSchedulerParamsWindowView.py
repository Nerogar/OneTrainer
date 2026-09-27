
from modules.ui.BaseConfigListView import BaseConfigListView


class BaseSchedulerParamsWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, master, controller, ui_state):
        if controller.is_custom_scheduler():
            self.components.label(master, 0, 0, "Class Name",
                                  tooltip="Python class module and name for the custom scheduler class, in the form of <module>.<class_name>.")
            self.components.entry(master, 0, 1, ui_state, "custom_learning_rate_scheduler")


class BaseKvParamsView(BaseConfigListView):
    def __init__(self, components):
        self.components = components

    def open_element_window(self, i, ui_state):
        pass
