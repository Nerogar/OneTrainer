import copy
import json
import os
from abc import abstractmethod, ABCMeta

import customtkinter as ctk

from modules.util import path_util
from modules.util.args.TrainArgs import TrainArgs
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState


class ConfigList(metaclass=ABCMeta):

    def __init__(
            self,
            master,
            train_args: TrainArgs,
            ui_state: UIState,
            element_attr_name: str,
            config_dir: str,
            default_config_name: str,
            add_button_text: str,
            is_full_width: bool,
    ):
        self.master = master
        self.train_args = train_args
        self.ui_state = ui_state
        self.element_attr_name = element_attr_name

        self.config_dir = config_dir
        self.default_config_name = default_config_name

        self.is_full_width = is_full_width

        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.top_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        self.top_frame.grid(row=0, column=0, sticky="nsew")

        self.configs_dropdown = None
        self.element_list = None

        self.configs = []
        self.__load_available_config_names()

        self.current_config = []
        self.widgets = []
        self.__load_current_config(getattr(self.train_args, self.element_attr_name))

        self.__create_configs_dropdown()

        components.icon_button(self.top_frame, 0, 2, "add config", self.__add_config)

        components.icon_button(self.top_frame, 0, 3, add_button_text, self.__add_element)

    @abstractmethod
    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        pass

    @abstractmethod
    def create_new_element(self) -> dict:
        pass

    @abstractmethod
    def open_element_window(self, i) -> ctk.CTkToplevel:
        pass

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.destroy()

        self.configs_dropdown = components.options_kv(
            self.top_frame, 0, 1, self.configs, self.ui_state, self.element_attr_name, self.__load_current_config
        )

    def __create_element_list(self):
        self.widgets = []
        if self.element_list is not None:
            self.element_list.destroy()

        self.element_list = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.element_list.grid(row=1, column=0, sticky="nsew")

        if self.is_full_width:
            self.element_list.grid_columnconfigure(0, weight=1)

        for i, element in enumerate(self.current_config):
            widget = self.create_widget(
                self.element_list, element, i,
                self.__open_element_window,
                self.__remove_element,
                self.__clone_element,
                self.__save_current_config
            )
            self.widgets.append(widget)

            widget.place_in_list()

    def __load_available_config_names(self):
        if os.path.isdir(self.config_dir):
            for path in os.listdir(self.config_dir):
                path = path_util.canonical_join(self.config_dir, path)
                if path.endswith(".json") and os.path.isfile(path):
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    self.configs.append((name, path))

        if len(self.configs) == 0:
            name = self.default_config_name.removesuffix(".json")
            self.__create_config(name)
            self.__save_current_config()

    def __create_config(self, name: str):
        name = path_util.safe_filename(name)
        path = path_util.canonical_join(self.config_dir, f"{name}.json")
        self.configs.append((name, path))
        self.__create_configs_dropdown()

    def __add_config(self):
        dialogs.StringInputDialog(self.master, "name", "Name", self.__create_config)

    def __add_element(self):
        i = len(self.current_config)
        new_element = self.create_new_element()

        self.current_config.append(new_element)
        widget = self.create_widget(
            self.element_list, new_element, i,
            self.__open_element_window,
            self.__remove_element,
            self.__clone_element,
            self.__save_current_config
        )
        self.widgets.append(widget)

        widget.place_in_list()

        self.__save_current_config()

    def __clone_element(self, clone_i):
        i = len(self.current_config)
        new_element = copy.deepcopy(self.current_config[clone_i])

        self.current_config.append(new_element)
        widget = self.create_widget(
            self.element_list, new_element, i,
            self.__open_element_window,
            self.__remove_element,
            self.__clone_element,
            self.__save_current_config
        )
        self.widgets.append(widget)

        widget.place_in_list()

        self.__save_current_config()

    def __remove_element(self, remove_i):
        self.current_config.pop(remove_i)
        self.widgets.pop(remove_i).destroy()

        for i, widget in enumerate(self.widgets):
            widget.i = i
            widget.place_in_list()

        self.__save_current_config()

    def __load_current_config(self, filename):
        try:
            with open(filename, "r") as f:
                self.current_config = []

                loaded_config = json.load(f)
                for element in loaded_config:
                    element = self.create_new_element() | element
                    self.current_config.append(element)
        except:
            self.current_config = []

        self.__create_element_list()

    def __save_current_config(self):
        try:
            if not os.path.exists(self.config_dir):
                os.mkdir(self.config_dir)

            with open(getattr(self.train_args, self.element_attr_name), "w") as f:
                json.dump(self.current_config, f, indent=4)
        except:
            pass

    def __open_element_window(self, i):
        window = self.open_element_window(i)
        self.master.wait_window(window)
        self.widgets[i].configure_element()
        self.__save_current_config()
