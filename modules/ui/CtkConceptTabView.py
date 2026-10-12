from tkinter import BooleanVar, StringVar

from modules.ui.BaseConceptTabView import BaseConceptTabView, BaseConceptWidgetView
from modules.ui.ConceptTabController import ConceptTabController
from modules.ui.CtkConceptWindowView import CtkConceptWindowView
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.util.ui import ctk_components
from modules.util.ui.ctk_validation import DebounceTimer
from modules.util.ui.CtkUIState import CtkUIState

import customtkinter as ctk


class CtkConceptTabView(CtkConfigListView, BaseConceptTabView):

    def __init__(self, master, controller: ConceptTabController, ui_state):
        # Pre-initialize before CtkConfigListView.__init__ because _reset_filters is
        # called during build() via options_kv's immediate update_var() call.
        self.search_var = StringVar()
        self.filter_var = StringVar(value="ALL")
        self.show_disabled_var = BooleanVar(value=True)

        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            from_external_file=True,
            attr_name="concept_file_name",
            config_dir="training_concepts",
            default_config_name="concepts.json",
            add_button_text="Add Concept",
            add_button_tooltip="Adds a new concept to the current config.",
            is_full_width=False,
            show_toggle_button=True,
        )
        self._toolbar = None
        self._toolbar_is_wrapped = False
        self._add_search_bar()
        self.top_frame.bind('<Configure>', lambda e: self._maybe_reposition_toolbar(e.width))

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state[0], ui_state[1], ui_state[2], CtkConceptWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return CtkConceptWidgetView(master, element, i, open_command, remove_command, clone_command, save_command, self.controller)

    def _add_search_bar(self):
        toolbar = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        toolbar.grid(row=0, column=4, columnspan=2, padx=10, sticky="ew")
        toolbar.grid_columnconfigure(2, weight=1)
        self._toolbar = toolbar

        ctk.CTkLabel(toolbar, text="Search:").grid(row=0, column=0, padx=(0, 5))
        self.search_var = StringVar()
        self.search_entry = ctk.CTkEntry(toolbar, textvariable=self.search_var,
                                         placeholder_text="Filter...", width=200)
        self.search_entry.grid(row=0, column=1)
        self._search_debouncer = DebounceTimer(self.search_entry, 300, lambda: self._update_filters())
        self.search_var.trace_add("write", lambda *_: self._search_debouncer.call())

        ctk.CTkLabel(toolbar, text="").grid(row=0, column=2, padx=5)

        ctk.CTkLabel(toolbar, text="Type:").grid(row=0, column=3, padx=(0, 5))
        self.filter_var = StringVar(value="ALL")
        ctk.CTkOptionMenu(toolbar, values=self._FILTER_TYPES,
                          variable=self.filter_var, command=lambda x: self._update_filters(),
                          width=150).grid(row=0, column=4)

        self.show_disabled_var = BooleanVar(value=True)
        self.show_disabled_checkbox = ctk.CTkCheckBox(toolbar, text="Show Disabled", variable=self.show_disabled_var,
                                                      command=self._update_filters, width=100)
        self.show_disabled_checkbox.grid(row=0, column=5, padx=(10, 0))
        self._refresh_show_disabled_text()

        ctk.CTkButton(toolbar, text="Clear", width=50,
                      command=self._reset_filters).grid(row=0, column=6, padx=(10, 0))

    def _maybe_reposition_toolbar(self, width):
        if not self._toolbar:
            return
        threshold = 1070
        want_wrapped = width < threshold
        if want_wrapped == self._toolbar_is_wrapped:
            return
        self._toolbar_is_wrapped = want_wrapped
        if want_wrapped:
            self._toolbar.grid_configure(row=1, column=0, columnspan=8, sticky="ew", padx=10)
        else:
            self._toolbar.grid_configure(row=0, column=4, columnspan=2, sticky="ew", padx=10)

    def _update_filters(self):
        self._create_element_list(search=self.search_var.get(),
                                  type=self.filter_var.get(),
                                  show_disabled=self.show_disabled_var.get())
        self._refresh_show_disabled_text()

    def _reset_filters(self):
        self.search_var.set("")
        self.filter_var.set("ALL")
        self.show_disabled_var.set(True)
        self._update_filters()

    def _refresh_show_disabled_text(self):
        try:
            disabled_count = sum(1 for c in getattr(self, 'current_config', []) if getattr(c, 'enabled', True) is False)
        except (AttributeError, TypeError):
            disabled_count = 0
        text = f"Show Disabled ({disabled_count})" if disabled_count > 0 else "Show Disabled"
        try:
            if getattr(self, 'show_disabled_checkbox', None):
                self.show_disabled_checkbox.configure(text=text)
        except (AttributeError, RuntimeError):
            pass


class CtkConceptWidgetView(BaseConceptWidgetView, ctk.CTkFrame):

    def __init__(self, master, concept, i, open_command, remove_command, clone_command, save_command, controller):
        ctk.CTkFrame.__init__(self, master=master, width=150, height=170, corner_radius=10, bg_color="transparent")
        BaseConceptWidgetView.__init__(self, ctk_components, concept)
        self.ui_state = CtkUIState(self, concept)
        self.image_ui_state = CtkUIState(self, concept.image)
        self.text_ui_state = CtkUIState(self, concept.text)
        self.i = i

        self.grid_rowconfigure(1, weight=1)

        self.image = ctk.CTkImage(
            light_image=self._get_preview_image(),
            size=(150, 150)
        )
        image_label = ctk.CTkLabel(master=self, text="", image=self.image, height=150, width=150)
        image_label.grid(row=0, column=0)

        self.name_label = self.components.label(self, 1, 0, self._get_display_name(), pad=5, wraplength=140)

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

        clone_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="+",
            corner_radius=2,
            fg_color="#00C000",
            command=lambda: clone_command(self.i, controller.randomize_seed),
        )
        clone_button.place(x=25, y=0)

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

    def configure_element(self):
        self.name_label.configure(text=self._get_display_name())
        self.image.configure(light_image=self._get_preview_image())
        self._clear_search_cache()

    def place_in_list(self):
        index = getattr(self, 'visible_index', self.i)
        x = index % 6
        y = index // 6
        self.grid(row=y, column=x, pady=5, padx=5)
