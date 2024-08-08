import os

from modules.ui.ConceptWindow import ConceptWindow
from modules.ui.ConfigList import ConfigList
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk
from PIL import Image


class ConceptTab(ConfigList):

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            from_external_file=True,
            attr_name="concept_file_name",
            config_dir="training_concepts",
            default_config_name="concepts.json",
            add_button_text="add concept",
            is_full_width=False,
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return ConceptWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return ConceptConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        return ConceptWindow(self.master, self.current_config[i], ui_state[0], ui_state[1], ui_state[2])


class ConceptWidget(ctk.CTkFrame):
    def __init__(self, master, concept, i, open_command, remove_command, clone_command, save_command):
        super().__init__(
            master=master, width=150, height=170, corner_radius=10, bg_color="transparent"
        )

        self.concept = concept
        self.ui_state = UIState(self, concept)
        self.image_ui_state = UIState(self, concept.image)
        self.text_ui_state = UIState(self, concept.text)
        self.i = i

        self.grid_rowconfigure(1, weight=1)

        # image
        self.image = ctk.CTkImage(
            light_image=self.__get_preview_image(),
            size=(150, 150)
        )
        image_label = ctk.CTkLabel(master=self, text="", image=self.image, height=150, width=150)
        image_label.grid(row=0, column=0)

        # name
        self.name_label = components.label(self, 1, 0, self.__get_display_name(), pad=5, wraplength=140)

        # close button
        close_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="X",
            corner_radius=2,
            fg_color="#C00000",
            command=lambda: remove_command(self.i),
        )
        close_button.place(x=0, y=0)

        # clone button
        clone_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="+",
            corner_radius=2,
            fg_color="#00C000",
            command=lambda: clone_command(self.i, self.__randomize_seed),
        )
        clone_button.place(x=25, y=0)

        # enabled switch
        enabled_switch = ctk.CTkSwitch(
            master=self,
            width=40,
            variable=self.ui_state.get_var("enabled"),
            text="",
            command=save_command,
        )
        enabled_switch.place(x=110, y=0)

        image_label.bind(
            "<Button-1>",
            lambda event: open_command(self.i, (self.ui_state, self.image_ui_state, self.text_ui_state))
        )

    def __randomize_seed(self, concept: ConceptConfig):
        concept.seed = ConceptConfig.default_values().seed
        return concept

    def __get_display_name(self):
        if self.concept.name:
            return self.concept.name
        if self.concept.path:
            return os.path.basename(self.concept.path)
        return ""

    def configure_element(self):
        self.name_label.configure(text=self.__get_display_name())

        self.image.configure(light_image=self.__get_preview_image())

    def __get_preview_image(self):
        preview_path = "resources/icons/icon.png"

        if os.path.isdir(self.concept.path):
            for path in os.scandir(self.concept.path):
                extension = os.path.splitext(path)[1]
                if path.is_file() \
                        and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png"):
                    preview_path = path_util.canonical_join(self.concept.path, path.name)
                    break

        image = Image.open(preview_path)
        size = min(image.width, image.height)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width - size) // 2 + size,
            (image.height - size) // 2 + size,
        ))
        return image.resize((150, 150), Image.Resampling.LANCZOS)

    def place_in_list(self):
        x = self.i % 6
        y = self.i // 6

        self.grid(row=y, column=x, pady=5, padx=5)
