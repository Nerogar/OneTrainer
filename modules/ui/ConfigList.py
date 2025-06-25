import contextlib
import copy
import json
import os
import traceback
from abc import ABCMeta, abstractmethod

from modules.util import path_util
from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.path_util import write_json_atomic
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class ConfigList(metaclass=ABCMeta):

    def __init__(
            self,
            master,
            train_config: TrainConfig,
            ui_state: UIState,
            from_external_file: bool,
            attr_name: str = "",
            enabled_attr_name: str = "enabled",
            config_dir: str = "",
            default_config_name: str = "",
            add_button_text: str = "",
            is_full_width: bool = "",
            can_disable_all: bool = False,
    ):
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state
        self.from_external_file = from_external_file
        self.attr_name = attr_name
        self.enabled_attr_name = enabled_attr_name

        self.config_dir = config_dir
        self.default_config_name = default_config_name

        self.is_full_width = is_full_width

        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        if self.from_external_file:
            self.top_frame = ctk.CTkFrame(self.master, fg_color="transparent")
            self.top_frame.grid(row=0, column=0, sticky="nsew")

            self.configs_dropdown = None
            self.element_list = None

            self.configs = []
            self.__load_available_config_names()

            self.current_config = getattr(self.train_config, self.attr_name)
            self.widgets = []
            self.__load_current_config(getattr(self.train_config, self.attr_name))

            self.__create_configs_dropdown()
            components.icon_button(self.top_frame, 0, 2, "add config", self.__add_config)
            components.icon_button(self.top_frame, 0, 3, add_button_text, self.__add_element)
        else:
            self.top_frame = ctk.CTkFrame(self.master, fg_color="transparent")
            self.top_frame.grid(row=0, column=0, sticky="nsew")
            components.icon_button(self.top_frame, 0, 3, add_button_text, self.__add_element)

            self.current_config = getattr(self.train_config, self.attr_name)

            self.element_list = None
            self._create_element_list()

        if can_disable_all:
            components.icon_button(self.top_frame, 0, 4, "disable all", self.disable_all)

    @abstractmethod
    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        pass

    @abstractmethod
    def create_new_element(self) -> BaseConfig:
        pass

    @abstractmethod
    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass

    def disable_all(self):
        if self.from_external_file:
            current_config = getattr(self.train_config, self.attr_name)
            try:
                for (name, file_path) in self.configs:
                    with open(file_path, "r+") as f:
                        loaded_config = json.load(f)
                        for item in loaded_config:
                            if isinstance(item, dict):
                                item[self.enabled_attr_name] = False
                            else:
                                setattr(item, self.enabled_attr_name, False)
                    write_json_atomic(
                        file_path,
                        loaded_config
                    )
                self.__load_current_config(current_config)
            except Exception:
                traceback.print_exc()
                print("Failed to disable all items in all configs")
        else:
            for config in self.current_config:
                setattr(config, self.enabled_attr_name, False)
            self._create_element_list()

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.destroy()

        self.configs_dropdown = components.options_kv(
            self.top_frame, 0, 1, self.configs, self.ui_state, self.attr_name, self.__load_current_config
        )

    def _create_element_list(self):
        if not self.from_external_file:
            self.current_config = getattr(self.train_config, self.attr_name)

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
                self.save_current_config
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
            self.save_current_config()

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
            self.save_current_config
        )
        self.widgets.append(widget)

        widget.place_in_list()

        self.save_current_config()

    def __clone_element(self, clone_i, modify_element_fun=None):
        i = len(self.current_config)
        new_element = copy.deepcopy(self.current_config[clone_i])

        if modify_element_fun is not None:
            new_element = modify_element_fun(new_element)

        self.current_config.append(new_element)
        widget = self.create_widget(
            self.element_list, new_element, i,
            self.__open_element_window,
            self.__remove_element,
            self.__clone_element,
            self.save_current_config
        )
        self.widgets.append(widget)

        widget.place_in_list()

        self.save_current_config()

    def __remove_element(self, remove_i):
        self.current_config.pop(remove_i)
        self.widgets.pop(remove_i).destroy()

        for i, widget in enumerate(self.widgets):
            widget.i = i
            widget.place_in_list()

        self.save_current_config()

    def __load_current_config(self, filename):
        try:
            with open(filename, "r") as f:
                self.current_config = []

                loaded_config_json = json.load(f)
                for element_json in loaded_config_json:
                    element = self.create_new_element().from_dict(element_json)
                    self.current_config.append(element)
        except Exception:
            self.current_config = []

        self._create_element_list()

    def save_current_config(self):
        if self.from_external_file:
            with contextlib.suppress(Exception):
                if not os.path.exists(self.config_dir):
                    os.mkdir(self.config_dir)

                write_json_atomic(
                    getattr(self.train_config, self.attr_name),
                    [element.to_dict() for element in self.current_config]
                )

    def __open_element_window(self, i, ui_state):
        window = self.open_element_window(i, ui_state)
        self.master.wait_window(window)
        self.widgets[i].configure_element()
        self.save_current_config()
