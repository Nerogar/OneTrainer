
from modules.ui.BaseTrainingTabView import BaseTrainingTabView
from modules.ui.CtkOffloadingWindowView import CtkOffloadingWindowView
from modules.ui.CtkOptimizerParamsWindowView import CtkOptimizerParamsWindowView
from modules.ui.CtkSchedulerParamsWindowView import CtkSchedulerParamsWindowView
from modules.ui.CtkTimestepDistributionWindowView import CtkTimestepDistributionWindowView
from modules.ui.TrainingTabController import TrainingTabController
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkTrainingTabView(BaseTrainingTabView):
    def __init__(self, master, controller: TrainingTabController, ui_state):
        BaseTrainingTabView.__init__(self, ctk_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state
        self.scroll_frame = None

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=1)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=1)

        column_0 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_0.grid(row=0, column=0, sticky="nsew")
        column_0.grid_columnconfigure(0, weight=1)

        column_1 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_1.grid(row=0, column=1, sticky="nsew")
        column_1.grid_columnconfigure(0, weight=1)

        column_2 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_2.grid(row=0, column=2, sticky="nsew")
        column_2.grid_columnconfigure(0, weight=1)

        callbacks = {
            'restore_optimizer': lambda *args: self.controller.restore_optimizer_config(self.ui_state),
            'open_optimizer_params': self._open_optimizer_params_window,
            'restore_scheduler': self._restore_scheduler_config,
            'open_scheduler_params': self._open_scheduler_params_window,
            'open_offloading': self._open_offloading_window,
            'open_timestep_distribution': self._open_timestep_distribution_window,
        }

        self.build(column_0, column_1, column_2, self.controller, self.ui_state, callbacks)

    def _restore_scheduler_config(self, variable):
        if not hasattr(self, 'lr_scheduler_adv_comp'):
            return
        state = "normal" if self.controller.is_custom_scheduler_value(variable) else "disabled"
        self.lr_scheduler_adv_comp.configure(state=state)

    def _open_optimizer_params_window(self):
        self.master.wait_window(self.controller.open_optimizer_params_window(self.master, self.ui_state, CtkOptimizerParamsWindowView))

    def _open_scheduler_params_window(self):
        self.master.wait_window(self.controller.open_scheduler_params_window(self.master, self.ui_state, CtkSchedulerParamsWindowView))

    def _open_timestep_distribution_window(self):
        self.master.wait_window(self.controller.open_timestep_distribution_window(self.master, self.ui_state, CtkTimestepDistributionWindowView))

    def _open_offloading_window(self):
        self.master.wait_window(self.controller.open_offloading_window(self.master, self.ui_state, CtkOffloadingWindowView))
