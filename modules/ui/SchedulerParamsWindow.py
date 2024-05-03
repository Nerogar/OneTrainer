import customtkinter as ctk

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.ui import components


class SchedulerParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, train_config: TrainConfig, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.pareant = parent
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

        components.button(self, 1, 0, "ok", command=self.on_window_close)
        self.main_frame(self.frame)

    def main_frame(self, master):
        if self.train_config.learning_rate_scheduler == LearningRateScheduler.CUSTOM:
            components.label(master, 0, 0, "Class Name",
                             tooltip="Python class module and name for the custom scheduler class.")

        # Any additional parameters, in key-value form.
