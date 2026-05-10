
from modules.ui.AdditionalEmbeddingsTabController import AdditionalEmbeddingsTabController
from modules.ui.BaseAdditionalEmbeddingsTabView import BaseAdditionalEmbeddingsTabView, BaseEmbeddingWidgetView
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState

import customtkinter as ctk


class CtkAdditionalEmbeddingsTabView(CtkConfigListView, BaseAdditionalEmbeddingsTabView):

    def __init__(self, master, controller: AdditionalEmbeddingsTabController, ui_state):
        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="additional_embeddings",
            enable_key="train",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True,
            show_toggle_button=True,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return CtkEmbeddingWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class CtkEmbeddingWidgetView(BaseEmbeddingWidgetView, ctk.CTkFrame):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        ctk.CTkFrame.__init__(self, master=master, corner_radius=10, bg_color="transparent")
        BaseEmbeddingWidgetView.__init__(self, ctk_components)

        self.element = element
        ui_state = CtkUIState(self, element)

        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.grid_columnconfigure(3, weight=1)
        top_frame.grid_columnconfigure(5, weight=1)

        bottom_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.grid_columnconfigure(7, weight=1)

        self.build_content(top_frame, bottom_frame, ui_state, i, save_command, remove_command, clone_command, controller)

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
