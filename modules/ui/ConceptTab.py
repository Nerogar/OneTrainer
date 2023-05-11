import json
import os

import customtkinter as ctk
from PIL import Image

from modules.ui.ConceptWindow import ConceptWindow
from modules.util import path_util
from modules.util.args.TrainArgs import TrainArgs
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState


class ConceptTab:

    def __init__(self, master, train_args: TrainArgs, ui_state: UIState):
        self.master = master
        self.train_args = train_args
        self.ui_state = ui_state

        self.dir = "training_concepts"

        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.top_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.top_frame.grid(row=0, column=0, sticky="nsew")

        self.configs_dropdown = None
        self.concept_list = None

        self.configs = []
        self.__load_available_config_names()

        self.current_config = []
        self.widgets = []
        self.__load_current_config(self.train_args.concept_file_name)

        self.__create_configs_dropdown()

        components.icon_button(self.top_frame, 0, 2, "add config", self.__add_config)

        components.icon_button(self.top_frame, 0, 3, "add concept", self.__add_concept)

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.destroy()

        self.configs_dropdown = components.options_kv(
            self.top_frame, 0, 1, self.configs, self.ui_state, "concept_file_name", self.__load_current_config
        )

    def __create_concept_list(self):
        self.widgets = []
        if self.concept_list is not None:
            self.concept_list.destroy()

        self.concept_list = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.concept_list.grid(row=1, column=0, sticky="nsew")

        for i, concept in enumerate(self.current_config):
            widget = ConceptWidget(self.concept_list, concept, i, self.__open_concept_window, self.__remove_concept)
            self.widgets.append(widget)

            widget.place_in_list()

    def __load_available_config_names(self):
        if os.path.isdir(self.dir):
            for path in os.listdir(self.dir):
                path = path_util.canonical_join(self.dir, path)
                if path.endswith(".json") and os.path.isfile(path):
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    self.configs.append((name, path))

        if len(self.configs) == 0:
            path = path_util.canonical_join(self.dir, "concepts.json")
            self.configs.append(("concepts", path))

    def __add_config(self):
        def create_config(name):
            name = path_util.safe_filename(name)
            path = path_util.canonical_join(self.dir, f"{name}.json")
            self.configs.append((name, path))
            self.__create_configs_dropdown()

        dialogs.StringInputDialog(self.master, "name", "Concepts Name", create_config)

    def __add_concept(self):
        i = len(self.current_config)
        new_concept = {
            "name": "",
            "path": "",
        }

        self.current_config.append(new_concept)
        widget = ConceptWidget(self.concept_list, new_concept, i, self.__open_concept_window, self.__remove_concept)
        self.widgets.append(widget)

        widget.place_in_list()

        self.__save_current_config()

    def __remove_concept(self, remove_i):
        self.current_config.pop(remove_i)
        self.widgets.pop(remove_i).destroy()

        for i, widget in enumerate(self.widgets):
            widget.i = i
            widget.place_in_list()

        self.__save_current_config()

    def __load_current_config(self, filename):
        try:
            with open(filename, "r") as f:
                self.current_config = json.load(f)
        except:
            self.current_config = []

        self.__create_concept_list()

    def __save_current_config(self):
        try:
            with open(self.train_args.concept_file_name, "w") as f:
                json.dump(self.current_config, f, indent=4)
        except:
            pass

    def __open_concept_window(self, i):
        window = ConceptWindow(self.master, self.current_config[i])
        self.master.wait_window(window)
        self.widgets[i].configure_concept()
        self.__save_current_config()


class ConceptWidget(ctk.CTkFrame):
    def __init__(self, master, concept, i, open_command, remove_command, **kwargs):
        super(ConceptWidget, self).__init__(
            master=master, width=150, height=170, corner_radius=10, bg_color="transparent"
        )

        self.concept = concept
        self.i = i
        self.command = open_command

        self.grid_rowconfigure(1, weight=1)

        # image
        self.image = ctk.CTkImage(
            light_image=self.__get_preview_image(),
            size=(150, 150)
        )
        image_label = ctk.CTkLabel(master=self, text="", image=self.image, height=150, width=150)
        image_label.grid(row=0, column=0)

        # name
        self.name_label = components.label(self, 1, 0, self.concept["name"], pad=5)

        # close button
        self.close_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="X",
            corner_radius=4,
            fg_color="#C00000",
            command=lambda: remove_command(self.i),
        )
        self.close_button.place(x=0, y=0)

        image_label.bind("<Button-1>", lambda event: self.command(self.i))

    def configure_concept(self):
        self.name_label.configure(text=self.concept["name"])

        self.image.configure(light_image=self.__get_preview_image())

    def __get_preview_image(self):
        preview_path = "resources/icons/icon.png"

        if os.path.isdir(self.concept["path"]):
            for path in os.scandir(self.concept["path"]):
                extension = os.path.splitext(path)[1]
                if path.is_file() \
                        and extension in path_util.supported_image_extensions() \
                        and not path.name.endswith("-masklabel.png"):
                    preview_path = path_util.canonical_join(self.concept["path"], path.name)
                    break

        image = Image.open(preview_path)
        size = min(image.width, image.height)
        image = image.crop((
            (image.width - size) // 2,
            (image.width - size) // 2,
            (image.width - size) // 2 + size,
            (image.width - size) // 2 + size,
        ))
        image = image.resize((150, 150), Image.Resampling.LANCZOS)
        return image

    def place_in_list(self):
        x = self.i % 6
        y = self.i // 6

        self.grid(row=y, column=x, pady=5, padx=5)
