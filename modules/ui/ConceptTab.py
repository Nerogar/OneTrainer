import os
import pathlib
from tkinter import BooleanVar, StringVar

from modules.ui.ConceptWindow import ConceptWindow
from modules.ui.ConfigList import ConfigList
from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_util import load_image
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
        self._add_search_bar()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return ConceptWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return ConceptConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        return ConceptWindow(self.master, self.current_config[i], ui_state[0], ui_state[1], ui_state[2])

    def _add_search_bar(self):
        """Add search and filter toolbar"""
        toolbar = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        toolbar.grid(row=0, column=4, columnspan=2, padx=10, sticky="ew")
        toolbar.grid_columnconfigure(2, weight=1)

        # Search
        ctk.CTkLabel(toolbar, text="Search:").grid(row=0, column=0, padx=(0,5))
        self.search_var = StringVar()
        self.search_entry = ctk.CTkEntry(toolbar, textvariable=self.search_var,
                                         placeholder_text="Filter...", width=200)
        self.search_entry.grid(row=0, column=1)
        self.search_var.trace("w", lambda *a: self._update_filters())

        # Spacer
        ctk.CTkLabel(toolbar, text="").grid(row=0, column=2, padx=5)

        # Type filter
        ctk.CTkLabel(toolbar, text="Type:").grid(row=0, column=3, padx=(0,5))
        self.filter_var = StringVar(value="ALL")
        ctk.CTkOptionMenu(toolbar, values=["ALL", "STANDARD", "VALIDATION", "PRIOR_PREDICTION"],
                          variable=self.filter_var, command=lambda x: self._update_filters(),
                          width=150).grid(row=0, column=4)

        # Show disabled checkbox
        self.show_disabled_var = BooleanVar(value=True)
        ctk.CTkCheckBox(toolbar, text="Show Disabled", variable=self.show_disabled_var,
                        command=self._update_filters, width=100).grid(row=0, column=5, padx=(10,0))

        # Clear button
        ctk.CTkButton(toolbar, text="Clear", width=50,
                      command=self._clear_filters).grid(row=0, column=6, padx=(10,0))

    def _update_filters(self):
        """Update view with current filters"""
        self._create_element_list(search=self.search_var.get(),
                                  type=self.filter_var.get(),
                                  show_disabled=self.show_disabled_var.get())

    def _clear_filters(self):
        """Reset all filters"""
        self.search_var.set("")
        self.filter_var.set("ALL")
        self.show_disabled_var.set(True)
        self._update_filters()

    def _element_matches_filters(self, element):
        """Combined filter matching"""
        # Check enabled status
        if not self.filters.get("show_disabled", True):
            if hasattr(element, 'enabled') and not element.enabled:
                return False

        # Search filter
        search = self.filters.get("search", "").lower()
        if search:
            texts = []
            if element.name:
                texts.append(element.name.lower())
            if element.path:
                texts.extend([os.path.basename(element.path).lower(),
                              element.path.lower()])
            if not any(search in text for text in texts):
                return False

        # Type filter
        type_filter = self.filters.get("type", "ALL")
        if type_filter != "ALL":
            if hasattr(element, 'type') and element.type:
                try:
                    return ConceptType(element.type).value == type_filter
                except (ValueError, AttributeError):
                    return True
            return False

        return True


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
        elif self.concept.path:
            return os.path.basename(self.concept.path)
        else:
            return ""

    def configure_element(self):
        self.name_label.configure(text=self.__get_display_name())

        self.image.configure(light_image=self.__get_preview_image())

    def __get_preview_image(self):
        preview_path = "resources/icons/icon.png"
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        if os.path.isdir(self.concept.path):
            for path in pathlib.Path(self.concept.path).glob(glob_pattern):
                extension = os.path.splitext(path)[1]
                if path.is_file() and path_util.is_supported_image_extension(extension) \
                        and not path.name.endswith("-masklabel.png") and not path.name.endswith("-condlabel.png"):
                    preview_path = path_util.canonical_join(self.concept.path, path)
                    break

        image = load_image(preview_path, convert_mode="RGBA")
        size = min(image.width, image.height)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width - size) // 2 + size,
            (image.height - size) // 2 + size,
        ))
        image = image.resize((150, 150), Image.Resampling.LANCZOS)
        return image

    def place_in_list(self):
        index = getattr(self, 'visible_index', self.i)
        x = index % 6
        y = index // 6

        self.grid(row=y, column=x, pady=5, padx=5)
