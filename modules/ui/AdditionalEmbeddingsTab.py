from pathlib import Path

from modules.ui.ConfigList import ConfigList
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class AdditionalEmbeddingsTab(ConfigList):

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="additional_embeddings",
            enable_key="train",
            from_external_file=False,
            add_button_text="add embedding",
            is_full_width=True,
            show_toggle_button=True
        )

    def refresh_ui(self):
        if self.element_list is not None:
            self.element_list.destroy()
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return EmbeddingWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return TrainEmbeddingConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass


class EmbeddingWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(
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
        bottom_frame.grid_columnconfigure(7, weight=1)

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
            command=lambda: clone_command(self.i, self.__randomize_uuid),
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
                         tooltip="The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.")
        token_count_entry = components.entry(top_frame, 0, 7, self.ui_state, "token_count")
        token_count_entry.configure(width=40)

        # trainable
        components.label(bottom_frame, 0, 0, "train:")
        trainable_switch = components.switch(bottom_frame, 0, 1, self.ui_state, "train", command=save_command)
        trainable_switch.configure(width=40)

        # output embedding
        components.label(bottom_frame, 0, 2, "output embedding:",
                         tooltip="Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.")
        output_embedding_switch = components.switch(bottom_frame, 0, 3, self.ui_state, "is_output_embedding")
        output_embedding_switch.configure(width=40)

        # stop training after
        components.label(bottom_frame, 0, 4, "stop training after:",
                         tooltip="When to stop training the embedding")
        components.time_entry(bottom_frame, 0, 5, self.ui_state, "stop_training_after", "stop_training_after_unit")

        # initial embedding text
        components.label(bottom_frame, 0, 6, "initial embedding text:",
                         tooltip="The initial embedding text used when creating a new embedding")
        components.entry(bottom_frame, 0, 7, self.ui_state, "initial_embedding_text")

    def __randomize_uuid(self, embedding_config: TrainEmbeddingConfig):
        embedding_config.uuid = TrainEmbeddingConfig.default_values().uuid
        return embedding_config

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
