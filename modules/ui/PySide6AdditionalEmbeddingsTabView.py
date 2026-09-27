from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseAdditionalEmbeddingsTabView import BaseAdditionalEmbeddingsTabView, BaseEmbeddingWidgetView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QWidget


class PySide6AdditionalEmbeddingsTabView(PySide6ConfigListView, BaseAdditionalEmbeddingsTabView):

    def __init__(self, master, controller: AdditionalEmbeddingsTabController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="additional_embeddings",
            enable_key="train",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True,
            show_toggle_button=True,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6EmbeddingWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class PySide6EmbeddingWidgetView(BaseEmbeddingWidgetView, QWidget):

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
