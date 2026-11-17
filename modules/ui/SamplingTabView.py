from abc import ABC, abstractmethod

from modules.ui.ConfigListView import BaseConfigListView, ConfigListView
from modules.ui.SampleParamsWindowView import SampleParamsWindowView
from modules.ui.SamplingTabController import SamplingTabController
from modules.util.enum.ModelType import ModelType
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class BaseSamplingTabView(BaseConfigListView):
    pass


class BaseSampleWidgetView(ABC):
    def __init__(self, components):
        self.components = components

    def build_content(self, frame, element, ui_state, i, open_command, remove_command, clone_command, save_command):
        self.element = element
        self.i = i
        self.save_command = save_command

        # close button
        self.components.list_action_button(frame, "remove", lambda: remove_command(self.i), row=0, column=0,
                                           tooltip="Deletes the sample")

        # clone button
        self.components.list_action_button(frame, "clone", lambda: clone_command(self.i), row=0, column=1,
                                           tooltip="Duplicates the sample")

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
        if self._has_clip_text_encoder:
            self.prompt_entry = self.components.entry(frame, 0, 10, ui_state, "prompt")
        else:
            self.prompt_entry = self.components.text_preview(frame, 0, 10, ui_state, "prompt", lines=2)

        # button
        self.button = self.components.icon_button(frame, 0, 11, "...", lambda: open_command(self.i, ui_state))

        self._bind_save(save_command)
        self._set_enabled()

    @abstractmethod
    def _bind_save(self, save_command): pass

    # BaseConfigListView calls configure_element() on all widget types generically;
    # sampling widgets have no post-window logic, so this is an intentional no-op.
    def configure_element(self): pass  # noqa: B027

    def _switch_enabled(self):
        self.save_command()
        self._set_enabled()

    @abstractmethod
    def _set_enabled(self):
        pass


class SamplingTabView(ConfigListView, BaseSamplingTabView):

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
        return SampleWidgetView(master, element, i, open_command, remove_command, clone_command, save_command,
                                model_type=self.controller.train_config.model_type)

    def refresh_ui(self):
        self.widgets_initialized = False
        self._create_element_list()


class SampleWidgetView(BaseSampleWidgetView, QWidget, metaclass=QtABCMeta):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command,
                 *, model_type: ModelType):
        QWidget.__init__(self, master)
        BaseSampleWidgetView.__init__(self, pyside6_components)

        from modules.util.ui.PySide6UIState import PySide6UIState
        self.element = element
        self.ui_state = PySide6UIState(element)
        self._has_clip_text_encoder = model_type.has_clip_text_encoder()

        pyside6_components._layout(self).setColumnStretch(10, 1)

        self.build_content(self, element, self.ui_state, i, open_command, remove_command, clone_command, save_command)

        layout = pyside6_components._layout(self)
        for index in range(layout.count()):
            item = layout.itemAt(index)
            layout.setAlignment(item.widget(),
                                (item.alignment() & Qt.AlignmentFlag.AlignHorizontal_Mask) | Qt.AlignmentFlag.AlignVCenter)

    def _bind_save(self, save_command):
        self.width_entry.editingFinished.connect(save_command)
        self.height_entry.editingFinished.connect(save_command)
        self.seed_entry.editingFinished.connect(save_command)
        if self._has_clip_text_encoder:
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
