import json
import math
import os
from tkinter import TclError

import customtkinter as ctk

from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import UserPreferenceUtility, OPTIMIZER_KEY_MAP
from modules.util.ui import components


class OptimizerParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)

        self.pref_util = UserPreferenceUtility()

        self.parent = parent

        self.train_args = TrainArgs.default_values()
        self.ui_state = ui_state
        self.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.title("Optimizer Settings")
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
        self.frame.grid_columnconfigure(2, minsize=50)
        self.frame.grid_columnconfigure(3, weight=0)
        self.frame.grid_columnconfigure(4, weight=1)

        components.button(self, 1, 0, "ok", command=self.on_window_close)
        self.button = None
        self.main_frame(self.frame)

    def on_window_close(self):
        selected_optimizer = self.ui_state.vars['optimizer'].get()
        for key in OPTIMIZER_KEY_MAP[selected_optimizer]:
            value = self.ui_state.vars[key].get()
            self.update_user_pref(selected_optimizer, key, value)

        self.destroy()

    def create_dynamic_ui(self, selected_optimizer, master, components, ui_state):

        # Lookup for the title and tooltip for a key
        # @formatter:off
        KEY_DETAIL_MAP = {
            'optimizer_adam_w_mode': {'title': 'Adam W Mode', 'tooltip': 'Whether to use weight decay correction for Adam optimizer.', 'type': 'bool'},
            'optimizer_alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'optimizer_amsgrad': {'title': 'AMSGrad', 'tooltip': 'Whether to use the AMSGrad variant for Adam.', 'type': 'bool'},
            'optimizer_beta1': {'title': 'Beta1', 'tooltip': 'optimizer_momentum term.', 'type': 'float'},
            'optimizer_beta2': {'title': 'Beta2', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'float'},
            'optimizer_beta3': {'title': 'Beta3', 'tooltip': 'Coefficient for computing the Prodigy stepsize.', 'type': 'float'},
            'optimizer_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Whether to use bias correction in optimization algorithms like Adam.', 'type': 'bool'},
            'optimizer_block_wise': {'title': 'Block Wise', 'tooltip': 'Whether to perform block-wise model update.', 'type': 'bool'},
            'optimizer_capturable': {'title': 'Capturable', 'tooltip': 'Whether some property of the optimizer can be captured.', 'type': 'bool'},
            'optimizer_centered': {'title': 'Centered', 'tooltip': 'Whether to center the gradient before scaling. Great for stabilizing the training process.', 'type': 'bool'},
            'optimizer_clip_threshold': {'title': 'Clip Threshold', 'tooltip': 'Clipping value for gradients.', 'type': 'float'},
            'optimizer_d0': {'title': 'Initial D', 'tooltip': 'Initial D estimate for D-adaptation.', 'type': 'float'},
            'optimizer_d_coef': {'title': 'D Coefficient', 'tooltip': 'Coefficient in the expression for the estimate of d.', 'type': 'float'},
            'optimizer_dampening': {'title': 'Dampening', 'tooltip': 'Dampening for optimizer_momentum.', 'type': 'float'},
            'optimizer_decay_rate': {'title': 'Decay Rate', 'tooltip': 'Rate of decay for moment estimation.', 'type': 'float'},
            'optimizer_decouple': {'title': 'Decouple', 'tooltip': 'Use AdamW style optimizer_decoupled weight decay.', 'type': 'bool'},
            'optimizer_differentiable': {'title': 'Differentiable', 'tooltip': 'Whether the optimization function is optimizer_differentiable.', 'type': 'bool'},
            'optimizer_eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'optimizer_eps2': {'title': 'EPS 2', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'optimizer_foreach': {'title': 'ForEach', 'tooltip': 'If true, apply the optimizer to each parameter independently.', 'type': 'bool'},
            'optimizer_fsdp_in_use': {'title': 'FSDP in Use', 'tooltip': 'Flag for using sharded parameters.', 'type': 'bool'},
            'optimizer_fused': {'title': 'Fused', 'tooltip': 'Whether to use a optimizer_fused implementation if available.', 'type': 'bool'},
            'optimizer_growth_rate': {'title': 'Growth Rate', 'tooltip': 'Limit for D estimate growth rate.', 'type': 'float'},
            'optimizer_initial_accumulator_value': {'title': 'Initial Accumulator Value', 'tooltip': 'Initial value for Adagrad optimizer.', 'type': 'float'},
            'optimizer_is_paged': {'title': 'Is Paged', 'tooltip': 'Whether the optimizer\'s internal state should be paged to CPU.', 'type': 'bool'},
            'optimizer_log_every': {'title': 'Log Every', 'tooltip': 'Intervals at which logging should occur.', 'type': 'int'},
            'optimizer_lr_decay': {'title': 'LR Decay', 'tooltip': 'Rate at which learning rate decreases.', 'type': 'float'},
            'optimizer_max_unorm': {'title': 'Max Unorm', 'tooltip': 'Maximum value for gradient clipping by norms.', 'type': 'float'},
            'optimizer_maximize': {'title': 'Maximize', 'tooltip': 'Whether to optimizer_maximize the optimization function.', 'type': 'bool'},
            'optimizer_min_8bit_size': {'title': 'Min 8bit Size', 'tooltip': 'Minimum tensor size for 8-bit quantization.', 'type': 'int'},
            'optimizer_momentum': {'title': 'optimizer_momentum', 'tooltip': 'Factor to accelerate SGD in relevant direction.', 'type': 'float'},
            'optimizer_nesterov': {'title': 'Nesterov', 'tooltip': 'Whether to enable Nesterov optimizer_momentum.', 'type': 'bool'},
            'optimizer_no_prox': {'title': 'No Prox', 'tooltip': 'Whether to use proximity updates or not.', 'type': 'bool'},
            'optimizer_optim_bits': {'title': 'Optim Bits', 'tooltip': 'Number of bits used for optimization.', 'type': 'int'},
            'optimizer_percentile_clipping': {'title': 'Percentile Clipping', 'tooltip': 'Gradient clipping based on percentile values.', 'type': 'float'},
            'optimizer_relative_step': {'title': 'Relative Step', 'tooltip': 'Whether to use a relative step size.', 'type': 'bool'},
            'optimizer_safeguard_warmup': {'title': 'Safeguard Warmup', 'tooltip': 'Avoid issues during warm-up stage.', 'type': 'bool'},
            'optimizer_scale_parameter': {'title': 'Scale Parameter', 'tooltip': 'Whether to scale the parameter or not.', 'type': 'bool'},
            'optimizer_use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'optimizer_use_triton': {'title': 'Use Triton', 'tooltip': 'Whether Triton optimization should be used.', 'type': 'bool'},
            'optimizer_warmup_init': {'title': 'Warmup Initialization', 'tooltip': 'Whether to warm-up the optimizer initialization.', 'type': 'bool'},
            'optimizer_weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
        }
        # @formatter:on

        if not self.winfo_exists():  # check if this window isn't open
            return

        # Extract the keys for the selected optimizer
        optimizer_keys = list(OPTIMIZER_KEY_MAP[selected_optimizer].keys())
        for idx, key in enumerate(optimizer_keys):
            arg_info = KEY_DETAIL_MAP[key]

            title = arg_info['title']
            tooltip = arg_info['tooltip']
            type = arg_info['type']

            row = math.floor(idx / 2) + 1
            col = 3 * (idx % 2)

            components.label(master, row, col, title, tooltip=tooltip)
            override_value = None

            user_prefs = self.pref_util.load_preferences(selected_optimizer)
            if user_prefs and key in user_prefs:
                override_value = user_prefs[key]
            elif key in OPTIMIZER_KEY_MAP[selected_optimizer]:
                override_value = OPTIMIZER_KEY_MAP[selected_optimizer][key]

            if type != 'bool':
                entry_widget = components.entry(master, row, col + 1, ui_state, key, override_value=override_value)
                entry_widget.bind("<FocusOut>", lambda event, opt=selected_optimizer, k=key: self.update_user_pref(opt, k, ui_state.vars[k].get()))
            else:
                switch_widget = components.switch(master, row, col + 1, ui_state, key, override_value=override_value)
                switch_widget.configure(command=lambda opt=selected_optimizer, k=key: self.update_user_pref(opt, k, ui_state.vars[k].get()))

    def update_user_pref(self, optimizer, key, value):
        self.pref_util.save_preference(optimizer, key, value)

    def remove_user_pref(self, optimizer):
        self.pref_util.remove_preference(optimizer)

    def main_frame(self, master):
        # Optimizer
        components.label(master, 0, 0, "Optimizer",
                         tooltip="The type of optimizer")

        # Create the optimizer dropdown menu and set the command
        optimizer_menu = components.options(master, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer",
                                             command=self.on_optimizer_change)

        # Defaults Button
        components.label(master, 0, 3, "Optimizer Defaults",
                         tooltip="Load default settings for the selected optimizer")
        components.button(self.frame, 0, 4, "Load Defaults", self.load_defaults,
                          tooltip="Load default settings for the selected optimizer")

        selected_optimizer = self.ui_state.vars['optimizer'].get()

        self.create_dynamic_ui(selected_optimizer, master, components, self.ui_state)

    def on_optimizer_change(self, *args):
        selected_optimizer = self.ui_state.vars['optimizer'].get()
        user_prefs = self.pref_util.load_preferences(selected_optimizer)

        for key, default_value in OPTIMIZER_KEY_MAP[selected_optimizer].items():
            if user_prefs == "Use_Default":
                value_to_set = default_value
            else:
                value_to_set = user_prefs.get(key, default_value)

            self.ui_state.vars[key].set(value_to_set)

        if not self.winfo_exists():  # check if this window isn't open
            return
        self.clear_dynamic_ui(self.frame)
        self.create_dynamic_ui(selected_optimizer, self.frame, components, self.ui_state)

    def load_defaults(self):
        if not self.winfo_exists():  # check if this window isn't open
            return
        selected_optimizer = self.ui_state.vars['optimizer'].get()
        self.clear_dynamic_ui(self.frame)
        self.remove_user_pref(selected_optimizer)
        self.create_dynamic_ui(selected_optimizer, self.frame, components, self.ui_state)

    def clear_dynamic_ui(self, master):
        try:
            for widget in master.winfo_children():
                grid_info = widget.grid_info()
                if int(grid_info["row"]) >= 1:
                    widget.destroy()
        except TclError as e:
            pass
