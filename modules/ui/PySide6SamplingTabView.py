from modules.ui.BaseSamplingTabView import BaseSampleWidgetView, BaseSamplingTabView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.ui.PySide6SampleParamsWindowView import PySide6SampleParamsWindowView
from modules.ui.SamplingTabController import SamplingTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QWidget


class PySide6SamplingTabView(PySide6ConfigListView, BaseSamplingTabView):

    def __init__(self, master, controller: SamplingTabController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            from_external_file=True,
            attr_name="sample_definition_file_name",
            config_dir="training_samples",
            default_config_name="samples.json",
            add_button_text="Add Sample",
            add_button_tooltip="Add a new sample configuration.",
            is_full_width=True,
            show_toggle_button=True,
        )

    def open_element_window(self, i, ui_state):
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state, PySide6SampleParamsWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6SampleWidgetView(master, element, i, open_command, remove_command, clone_command, save_command)


class PySide6SampleWidgetView(BaseSampleWidgetView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        QWidget.__init__(self, master)
        BaseSampleWidgetView.__init__(self, pyside6_components)

        from modules.util.ui.PySide6UIState import PySide6UIState
        self.element = element
        self.ui_state = PySide6UIState(element)

        pyside6_components._layout(self).setColumnStretch(10, 1)

        self.build_content(self, element, self.ui_state, i, open_command, remove_command, clone_command, save_command)

    def _bind_save(self, save_command):
        self.width_entry.editingFinished.connect(save_command)
        self.height_entry.editingFinished.connect(save_command)
        self.seed_entry.editingFinished.connect(save_command)
        self.prompt_entry.editingFinished.connect(save_command)

    def _set_enabled(self):
        enabled = self.element.enabled
        self.width_entry.setEnabled(enabled)
        self.height_entry.setEnabled(enabled)
        self.prompt_entry.setEnabled(enabled)
        self.seed_entry.setEnabled(enabled)
        self.button.setEnabled(enabled)

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()
