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
            from_external_file=False,
            add_button_text="新增嵌入",
            is_full_width=True,
        )

    def refresh_ui(self):
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
            command=lambda: clone_command(self.i, self.__randomize_uuid),
        )
        clone_button.grid(row=0, column=1, padx=5)

        # embedding model names
        components.label(top_frame, 0, 2, "基础嵌入：",
                         tooltip="要训练的基础嵌入。留空以创建新的嵌入")
        components.file_entry(
            top_frame, 0, 3, self.ui_state, "model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # placeholder
        components.label(top_frame, 0, 4, "触发词：",
                         tooltip="在提示中使用嵌入时使用的触发词")
        components.entry(top_frame, 0, 5, self.ui_state, "placeholder")

        # token count
        components.label(top_frame, 0, 6, "令牌计数：",
                         tooltip="创建新嵌入时使用的令牌计数")
        token_count_entry = components.entry(top_frame, 0, 7, self.ui_state, "token_count")
        token_count_entry.configure(width=40)

        # trainable
        components.label(bottom_frame, 0, 0, "训练:")
        trainable_switch = components.switch(bottom_frame, 0, 1, self.ui_state, "train")
        trainable_switch.configure(width=40)

        # stop training after
        components.label(bottom_frame, 0, 2, "在以下情况下停止训练：",
                         tooltip="何时停止训练嵌入")
        components.time_entry(bottom_frame, 0, 3, self.ui_state, "stop_training_after", "stop_training_after_unit")

        # initial embedding text
        components.label(bottom_frame, 0, 4, "初始嵌入文本：",
                         tooltip="创建新嵌入时使用的初始嵌入文本")
        components.entry(bottom_frame, 0, 5, self.ui_state, "initial_embedding_text")

    def __randomize_uuid(self, embedding_config: TrainEmbeddingConfig):
        embedding_config.uuid = TrainEmbeddingConfig.default_values().uuid
        return embedding_config

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
