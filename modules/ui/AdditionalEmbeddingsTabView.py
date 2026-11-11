
from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.ConfigListView import BaseConfigListView, ConfigListView
from modules.util import path_util
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QWidget


class BaseAdditionalEmbeddingsTabView(BaseConfigListView):

    def refresh_ui(self):
        if self.element_list is not None:
            self._destroy_frame(self.element_list)
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()

    def open_element_window(self, i, ui_state):
        pass


class BaseEmbeddingWidgetView:

    def __init__(self, components):
        self.components = components

    def build_content(self, top_frame, bottom_frame, ui_state, i, save_command, remove_command, clone_command, controller):
        self.ui_state = ui_state
        self.i = i
        self.save_command = save_command

        # close button
        self.components.colored_icon_button(top_frame, 0, 0, "X", "#C00000", lambda: remove_command(self.i))

        # clone button
        self.components.colored_icon_button(top_frame, 0, 1, "+", "#00C000", lambda: clone_command(self.i, controller.randomize_uuid), padx=5)

        # embedding model names
        self.components.label(top_frame, 0, 2, "base embedding:",
                              tooltip="The base embedding to train on. Leave empty to create a new embedding")
        self.components.path_entry(
            top_frame, 0, 3, self.ui_state, "model_name",
            mode="file", path_modifier=path_util.json_path_modifier
        )

        # placeholder
        self.components.label(top_frame, 0, 4, "placeholder:",
                              tooltip="The placeholder used when using the embedding in a prompt")
        self.components.entry(top_frame, 0, 5, self.ui_state, "placeholder")

        # token count
        self.components.label(top_frame, 0, 6, "token count:",
                              tooltip="The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.")
        self.components.entry(top_frame, 0, 7, self.ui_state, "token_count", width=40)

        # trainable
        self.components.label(bottom_frame, 0, 0, "train:")
        self.components.switch(bottom_frame, 0, 1, self.ui_state, "train", command=save_command, width=40)

        # output embedding
        self.components.label(bottom_frame, 0, 2, "output embedding:",
                              tooltip="Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.")
        self.components.switch(bottom_frame, 0, 3, self.ui_state, "is_output_embedding", width=40)

        # stop training after
        self.components.label(bottom_frame, 0, 4, "stop training after:",
                              tooltip="When to stop training the embedding")
        self.components.time_entry(bottom_frame, 0, 5, self.ui_state, "stop_training_after", "stop_training_after_unit")

        # initial embedding text
        self.components.label(bottom_frame, 0, 6, "initial embedding text:",
                              tooltip="The initial embedding text used when creating a new embedding")
        self.components.entry(bottom_frame, 0, 7, self.ui_state, "initial_embedding_text")

    def configure_element(self):
        pass


class AdditionalEmbeddingsTabView(ConfigListView, BaseAdditionalEmbeddingsTabView):

    def __init__(self, master, controller: AdditionalEmbeddingsTabController, ui_state):
        ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="additional_embeddings",
            enable_key="train",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True,
            show_toggle_button=True,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return EmbeddingWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class EmbeddingWidgetView(BaseEmbeddingWidgetView, QWidget):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        QWidget.__init__(self, master)
        BaseEmbeddingWidgetView.__init__(self, pyside6_components)

        self.element = element
        ui_state = PySide6UIState(element)

        pyside6_components._layout(self).setColumnStretch(0, 1)

        top_frame = QWidget(self)
        pyside6_components._layout(top_frame).setColumnStretch(3, 1)
        pyside6_components._layout(top_frame).setColumnStretch(5, 1)
        pyside6_components._layout(self).addWidget(top_frame, 0, 0)

        bottom_frame = QWidget(self)
        pyside6_components._layout(bottom_frame).setColumnStretch(7, 1)
        pyside6_components._layout(self).addWidget(bottom_frame, 1, 0)

        self.build_content(top_frame, bottom_frame, ui_state, i, save_command, remove_command, clone_command, controller)

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()
