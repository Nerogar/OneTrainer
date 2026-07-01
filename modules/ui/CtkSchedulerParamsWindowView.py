from modules.ui.BaseSchedulerParamsWindowView import BaseKvParamsView, BaseSchedulerParamsWindowView
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.ui.SchedulerParamsWindowController import KvParamsController, SchedulerParamsWindowController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkKvParamsView(CtkConfigListView, BaseKvParamsView):
    def __init__(self, master, controller: KvParamsController, ui_state):
        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="scheduler_params",
            from_external_file=False,
            add_button_text="add parameter",
            is_full_width=True,
        )
        BaseKvParamsView.__init__(self, ctk_components)

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return KvWidget(master, element, i, open_command, remove_command, clone_command, save_command)


class KvWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master=master, bg_color="transparent")
        self.element = element
        self.ui_state = CtkUIState(self, element)
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
        self.key = ctk_components.entry(self, 0, 1, self.ui_state, "key",
                                    tooltip=tooltip_key, wide_tooltip=True)
        self.key.bind("<FocusOut>", lambda _: save_command())
        self.key.configure(width=50)

        # Value
        tooltip_val = "Value for an argument in your scheduler. Some special values can be used, wrapped in percent signs: LR, EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS, SCHEDULER_STEPS. Note that OneTrainer calls step() after every individual learning step, not every epoch, so what Torch calls 'epoch' you should treat as 'step'."
        self.value = ctk_components.entry(self, 0, 2, self.ui_state, "value",
                                      tooltip=tooltip_val, wide_tooltip=True)
        self.value.bind("<FocusOut>", lambda _: save_command())
        self.value.configure(width=50)

    def place_in_list(self):
        self.grid(row=self.i, column=0, padx=5, pady=5, sticky="new")


class CtkSchedulerParamsWindowView(BaseSchedulerParamsWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: SchedulerParamsWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseSchedulerParamsWindowView.__init__(self, ctk_components)

        self.title("Learning Rate Scheduler Settings")
        self.geometry("800x400")
        self.resizable(True, True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)

        expand_frame = ctk.CTkFrame(frame, bg_color="transparent")
        expand_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.components.button(self, 1, 0, "ok", command=self.destroy)
        self.build_content(frame, controller, ui_state)
        CtkKvParamsView(expand_frame, KvParamsController(controller.config), ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
