from modules.ui.ConfigList import ConfigList
from modules.ui.SampleParamsWindow import SampleParamsWindow
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class SamplingTab(ConfigList):

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            from_external_file=True,
            attr_name="sample_definition_file_name",
            config_dir="training_samples",
            default_config_name="samples.json",
            add_button_text="Add Sample",
            add_button_tooltip="Add a new sample configuration.",
            is_full_width=True,
            show_toggle_button=True
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return SampleWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return SampleConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        return SampleParamsWindow(self.master, self.current_config[i], ui_state)


class SampleWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(
            master=master, corner_radius=10, bg_color="transparent"
        )

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(10, weight=1)

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
        close_button.grid(row=0, column=0)

        # clone button
        clone_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="+",
            corner_radius=2,
            fg_color="#00C000",
            command=lambda: clone_command(self.i),
        )
        clone_button.grid(row=0, column=1, padx=5)

        # enabled
        self.enabled_switch = components.switch(self, 0, 2, self.ui_state, "enabled", self.__switch_enabled)
        self.enabled_switch.configure(width=40)

        # width
        components.label(self, 0, 3, "width:")
        self.width_entry = components.entry(self, 0, 4, self.ui_state, "width")
        self.width_entry.bind('<FocusOut>', lambda _: save_command())
        self.width_entry.configure(width=50)

        # height
        components.label(self, 0, 5, "height:")
        self.height_entry = components.entry(self, 0, 6, self.ui_state, "height")
        self.height_entry.bind('<FocusOut>', lambda _: save_command())
        self.height_entry.configure(width=50)

        # seed
        components.label(self, 0, 7, "seed:")
        self.seed_entry = components.entry(self, 0, 8, self.ui_state, "seed")
        self.seed_entry.bind('<FocusOut>', lambda _: save_command())
        self.seed_entry.configure(width=80)

        # prompt
        components.label(self, 0, 9, "prompt:")
        self.prompt_entry = components.entry(self, 0, 10, self.ui_state, "prompt")
        self.prompt_entry.bind('<FocusOut>', lambda _: save_command())

        # button
        self.button = components.icon_button(self, 0, 11, "...", lambda: open_command(self.i, self.ui_state))
        self.button.configure(width=40)

        self.__set_enabled()

    def __switch_enabled(self):
        self.save_command()
        self.__set_enabled()

    def __set_enabled(self):
        enabled = self.element.enabled
        self.width_entry.configure(state="normal" if enabled else "disabled")
        self.height_entry.configure(state="normal" if enabled else "disabled")
        self.prompt_entry.configure(state="normal" if enabled else "disabled")
        self.seed_entry.configure(state="normal" if enabled else "disabled")
        self.button.configure(state="normal" if enabled else "disabled")

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
