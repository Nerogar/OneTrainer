import customtkinter as ctk

from modules.ui.ConfigList import ConfigList
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class KvParams(ConfigList):
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super(KvParams, self).__init__(
            master,
            train_config,
            ui_state,
            attr_name="scheduler_params",
            from_external_file=False,
            add_button_text="add parameter",
            is_full_width=True
        )

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return KvWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict[str, str]:
        return {"key": "", "value": ""}

    def open_element_window(self, i, ui_state):
        pass


class KvWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super(KvWidget, self).__init__(master=master, bg_color="transparent")
        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1, uniform=1)
        self.grid_columnconfigure(2, weight=1, uniform=1)

        close_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="X",
            corner_radius=2,
            fg_color="#C00000",
            command=lambda: remove_command(self.i))
        close_button.grid(row=0, column=0)

        # Key
        tooltip_key = "Key name for an argument in your scheduler"
        self.key = components.entry(self, 0, 1, self.ui_state, "key",
                                    tooltip=tooltip_key, wide_tooltip=True)
        self.key.bind("<FocusOut>", lambda _: save_command())
        self.key.configure(width=50)

        # Value
        tooltip_val = "Value for an argument in your scheduler. Some special values can be used, wrapped in percent signs: LR, EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS, SCHEDULER_STEPS. Note that OneTrainer calls step() after every individual learning step, not every epoch, so what Torch calls 'epoch' you should treat as 'step'."
        self.value = components.entry(self, 0, 2, self.ui_state, "value",
                                      tooltip=tooltip_val, wide_tooltip=True)
        self.value.bind("<FocusOut>", lambda _: save_command())
        self.value.configure(width=50)

    def place_in_list(self):
        self.grid(row=self.i, column=0, padx=5, pady=5, sticky="new")


class SchedulerParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, train_config: TrainConfig, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

        self.title("Learning Rate Scheduler Settings")
        self.geometry("800x400")
        self.resizable(True, True)
        self.wait_visibility()
        self.grab_set()
        self.focus_set()

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.frame = ctk.CTkFrame(self)
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.expand_frame = ctk.CTkFrame(self.frame, bg_color="transparent")
        self.expand_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        components.button(self, 1, 0, "ok", command=self.on_window_close)
        self.main_frame(self.frame)

    def main_frame(self, master):
        if self.train_config.learning_rate_scheduler is LearningRateScheduler.CUSTOM:
            components.label(master, 0, 0, "Class Name",
                             tooltip="Python class module and name for the custom scheduler class, in the form of <module>.<class_name>.")
            components.entry(master, 0, 1, self.ui_state, "custom_learning_rate_scheduler")

        # Any additional parameters, in key-value form.
        self.params = KvParams(self.expand_frame, self.train_config, self.ui_state)

    def on_window_close(self):
        self.destroy()
