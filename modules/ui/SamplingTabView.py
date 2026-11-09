from modules.ui.ConfigListView import ConfigListView
from modules.ui.SampleParamsWindowView import SampleParamsWindowView
from modules.ui.SamplingTabController import SamplingTabController
from modules.util.ui import pyside6_components

from PySide6.QtWidgets import QWidget


class SamplingTabView(ConfigListView):

    def __init__(self, master, controller: SamplingTabController, ui_state):
        ConfigListView.__init__(
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
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state, SampleParamsWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return SampleWidgetView(master, element, i, open_command, remove_command, clone_command, save_command)


class SampleWidgetView(QWidget):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        QWidget.__init__(self, master)
        self.components = pyside6_components

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

    def build_content(self, frame, element, ui_state, i, open_command, remove_command, clone_command, save_command):
        self.element = element
        self.i = i
        self.save_command = save_command

        # close button
        self.components.colored_icon_button(frame, 0, 0, "X", "#C00000", lambda: remove_command(self.i))

        # clone button
        self.components.colored_icon_button(frame, 0, 1, "+", "#00C000", lambda: clone_command(self.i), padx=5)

        # enabled
        self.enabled_switch = self.components.switch(frame, 0, 2, ui_state, "enabled", self._switch_enabled, width=40)

        # width
        self.components.label(frame, 0, 3, "width:")
        self.width_entry = self.components.entry(frame, 0, 4, ui_state, "width", width=50)

        # height
        self.components.label(frame, 0, 5, "height:")
        self.height_entry = self.components.entry(frame, 0, 6, ui_state, "height", width=50)

        # seed
        self.components.label(frame, 0, 7, "seed:")
        self.seed_entry = self.components.entry(frame, 0, 8, ui_state, "seed", width=80)

        # prompt
        self.components.label(frame, 0, 9, "prompt:")
        self.prompt_entry = self.components.entry(frame, 0, 10, ui_state, "prompt")

        # button
        self.button = self.components.icon_button(frame, 0, 11, "...", lambda: open_command(self.i, ui_state))

        self._bind_save(save_command)
        self._set_enabled()

    # ConfigListView calls configure_element() on all widget types generically;
    # sampling widgets have no post-window logic, so this is an intentional no-op.
    def configure_element(self): pass  # noqa: B027

    def _switch_enabled(self):
        self.save_command()
        self._set_enabled()
