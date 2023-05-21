import customtkinter as ctk

from modules.ui.ConfigList import ConfigList
from modules.util.args.TrainArgs import TrainArgs
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SamplingTab(ConfigList):

    def __init__(self, master, train_args: TrainArgs, ui_state: UIState):
        super(SamplingTab, self).__init__(
            master, train_args, ui_state, "sample_definition_file_name",
            "training_samples", "samples.json", "add sample"
        )

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return SampleWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return {
            "prompt": "",
            "height": 512,
            "width": 512,
            "seed": 42,
        }

    def open_element_window(self, i) -> ctk.CTkToplevel:
        pass


class SampleWidget(ctk.CTkFrame):
    def __init__(self, master, sample, i, open_command, remove_command, clone_command, save_command):
        super(SampleWidget, self).__init__(
            master=master, corner_radius=10, bg_color="transparent"
        )

        self.grid_rowconfigure(8, weight=1)

        self.ui_state = UIState(self, sample)

        self.sample = sample
        self.i = i
        self.command = open_command

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

        # height
        components.label(self, 0, 2, "height:")
        height_entry = components.entry(self, 0, 3, self.ui_state, "height")
        height_entry.bind('<FocusOut>', lambda _: save_command())
        height_entry.configure(width=50)

        # width
        components.label(self, 0, 4, "width:")
        width_entry = components.entry(self, 0, 5, self.ui_state, "width")
        width_entry.bind('<FocusOut>', lambda _: save_command())
        width_entry.configure(width=50)

        # seed
        components.label(self, 0, 6, "seed:")
        seed_entry = components.entry(self, 0, 7, self.ui_state, "seed")
        seed_entry.bind('<FocusOut>', lambda _: save_command())
        seed_entry.configure(width=80)

        # prompt
        components.label(self, 0, 8, "prompt:")
        prompt_entry = components.entry(self, 0, 9, self.ui_state, "prompt")
        prompt_entry.bind('<FocusOut>', lambda _: save_command())

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
