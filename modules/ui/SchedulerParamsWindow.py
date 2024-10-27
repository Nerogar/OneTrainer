from modules.ui.ConfigList import ConfigList
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class KvParams(ConfigList):
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
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
        super().__init__(master=master, bg_color="transparent")
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
        tooltip_key = "调度程序中参数的键名"
        self.key = components.entry(self, 0, 1, self.ui_state, "key",
                                    tooltip=tooltip_key, wide_tooltip=True)
        self.key.bind("<FocusOut>", lambda _: save_command())
        self.key.configure(width=50)

        # Value
        tooltip_val = "调度程序中参数的值。可以使用一些特殊值，用百分号包裹：LR、EPOCHS、STEPS_PER_EPOCH、TOTAL_STEPS、SCHEDUER_STEPS。请注意，OneTrainer在每个单独的学习步骤之后调用step（），而不是在每个纪元之后，所以Torch所称的“纪元”应该被视为“步骤”。"
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

        self.title("学习率调度器设置")
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
            components.label(master, 0, 0, "自定义类名",
                             tooltip="自定义调度器类的 Python 类模块和名称，格式为 <模块>.<类名>。")
            components.entry(master, 0, 1, self.ui_state, "custom_learning_rate_scheduler")

        # Any additional parameters, in key-value form.
        self.params = KvParams(self.expand_frame, self.train_config, self.ui_state)

    def on_window_close(self):
        self.destroy()
