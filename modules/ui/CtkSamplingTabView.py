from modules.ui.BaseSamplingTabView import BaseSampleWidgetView, BaseSamplingTabView
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.ui.CtkSampleParamsWindowView import CtkSampleParamsWindowView
from modules.ui.SamplingTabController import SamplingTabController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState

import customtkinter as ctk


class CtkSamplingTabView(CtkConfigListView, BaseSamplingTabView):
    def __init__(self, master, controller: SamplingTabController, ui_state):
        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            from_external_file=True,
            attr_name="sample_definition_file_name",
            config_dir="training_samples",
            default_config_name="samples.json",
            add_button_text="Add Sample",
            add_button_tooltip="Add a new sample configuration.",
            is_full_width=True,
            show_toggle_button=True,
        )

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        return self.controller.open_element_window(self.master, self.current_config[i], ui_state, CtkSampleParamsWindowView)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return CtkSampleWidgetView(master, element, i, open_command, remove_command, clone_command, save_command)


class CtkSampleWidgetView(BaseSampleWidgetView, ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        ctk.CTkFrame.__init__(self, master=master, corner_radius=10, bg_color="transparent")
        BaseSampleWidgetView.__init__(self, ctk_components)

        self.ui_state = CtkUIState(self, element)

        self.grid_columnconfigure(10, weight=1)

        self.build_content(self, element, self.ui_state, i, open_command, remove_command, clone_command, save_command)

    def _bind_save(self, save_command):
        self.width_entry.bind('<FocusOut>', lambda _: save_command())
        self.height_entry.bind('<FocusOut>', lambda _: save_command())
        self.seed_entry.bind('<FocusOut>', lambda _: save_command())
        self.prompt_entry.bind('<FocusOut>', lambda _: save_command())

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
