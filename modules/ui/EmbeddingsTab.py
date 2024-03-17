from pathlib import Path

import customtkinter as ctk

from modules.ui.ConfigList import ConfigList
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class EmbeddingsTab(ConfigList):

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super(EmbeddingsTab, self).__init__(
            master,
            train_config,
            ui_state,
            attr_name="embeddings",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return EmbeddingWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return TrainEmbeddingConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass


class EmbeddingWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super(EmbeddingWidget, self).__init__(
            master=master, corner_radius=10, bg_color="transparent"
        )

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.grid_columnconfigure(3, weight=1)
        top_frame.grid_columnconfigure(5, weight=1)

        bottom_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.grid_columnconfigure(5, weight=1)

        # close button
        close_button = ctk.CTkButton(
            master=top_frame,
            width=20,
            height=20,
            text="X",
            corner_radius=2,
            fg_color="#C00000",
            command=lambda: remove_command(self.i),
        )
        close_button.grid(row=0, column=0)

        # clone button
        clone_button = ctk.CTkButton(
            master=top_frame,
            width=20,
            height=20,
            text="+",
            corner_radius=2,
            fg_color="#00C000",
            command=lambda: clone_command(self.i),
        )
        clone_button.grid(row=0, column=1, padx=5)

        # embedding model names
        components.label(top_frame, 0, 2, "base embedding:",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        components.file_entry(
            top_frame, 0, 3, self.ui_state, "model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # placeholder
        components.label(top_frame, 0, 4, "placeholder:",
                         tooltip="The placeholder used when using the embedding in a prompt")
        components.entry(top_frame, 0, 5, self.ui_state, "placeholder")

        # token count
        components.label(top_frame, 0, 6, "token count:",
                         tooltip="The token count used when creating a new embedding")
        token_count_entry = components.entry(top_frame, 0, 7, self.ui_state, "token_count")
        token_count_entry.configure(width=40)

        # trainable
        components.label(bottom_frame, 0, 0, "train:")
        trainable_switch = components.switch(bottom_frame, 0, 1, self.ui_state, "train")
        trainable_switch.configure(width=40)

        # stop training after
        components.label(bottom_frame, 0, 2, "stop training after:",
                         tooltip="When to stop training the embedding")
        components.time_entry(bottom_frame, 0, 3, self.ui_state, "stop_training_after", "stop_training_after_unit")

        # initial embedding text
        components.label(bottom_frame, 0, 4, "initial embedding text:",
                         tooltip="The initial embedding text used when creating a new embedding")
        components.entry(bottom_frame, 0, 5, self.ui_state, "initial_embedding_text")

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
