from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk

MUON_AUX_ADAM_DEFAULTS = {
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
}

class MuonAdamWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            train_config: TrainConfig,
            ui_state: UIState,
            parent_optimizer_type: Optimizer,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.adam_ui_state = ui_state
        self.parent_optimizer_type = parent_optimizer_type

        if self.parent_optimizer_type == Optimizer.MUON:
            self.title("Muon's Auxiliary AdamW Settings")
            self.adam_params_def = MUON_AUX_ADAM_DEFAULTS
        else:
            self.title("Muon_adv's Auxiliary AdamW_adv Settings")
            self.adam_params_def = OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.ADAMW_ADV]

        self.geometry("800x500")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, minsize=50)
        self.frame.grid_columnconfigure(3, weight=0)
        self.frame.grid_columnconfigure(4, weight=1)

        components.button(self, 1, 0, "ok", command=self.destroy)
        self.create_adam_params_ui(self.frame)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def create_adam_params_ui(self, master):
        # This is a large map, copied from OptimizerParamsWindow for simplicity.
        # @formatter:off
        KEY_DETAIL_MAP = {
            'alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'beta1': {'title': 'Beta1', 'tooltip': 'optimizer_momentum term.', 'type': 'float'},
            'beta2': {'title': 'Beta2', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'float'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'stochastic_rounding': {'title': 'Stochastic Rounding', 'tooltip': 'Stochastic rounding for weight updates. Improves quality when using bfloat16 weights.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
            'use_orthograd': {'title': 'use_orthograd', 'tooltip': 'Use orthograd method', 'type': 'bool'},
            'nnmf_factor': {'title': 'Factored Optimizer', 'tooltip': 'Enables a memory-efficient mode by applying fast low-rank factorization to the optimizers states. It combines factorization for magnitudes with 1-bit compression for signs, drastically reducing VRAM usage and allowing for larger models or batch sizes. This is an approximation which may slightly alter training dynamics.', 'type': 'bool'},
            'orthogonal_gradient': {'title': 'OrthoGrad', 'tooltip': 'Reduces overfitting by removing the gradient component parallel to the weight, thus improving generalization.', 'type': 'bool'},
            'use_atan2': {'title': 'Atan2 Scaling', 'tooltip': 'A robust replacement for eps, which also incorporates gradient clipping, bounding and stabilizing the optimizer updates.', 'type': 'bool'},
            'cautious_mask': {'title': 'Cautious Variant', 'tooltip': 'Applies a mask to dampen or zero-out momentum components that disagree with the current gradients direction.', 'type': 'bool'},
            'grams_moment': {'title': 'GRAMS Variant', 'tooltip': 'Aligns the momentum direction with the current gradient direction while preserving its accumulated magnitude.', 'type': 'bool'},
            'use_AdEMAMix': {'title': 'AdEMAMix EMA', 'tooltip': 'Adds a second, slow-moving EMA, which is combined with the primary momentum to stabilize updates, and accelerate the training.', 'type': 'bool'},
            'beta3_ema': {'title': 'Beta3 EMA', 'tooltip': 'Coefficient for slow-moving EMA of AdEMAMix.', 'type': 'float'},
            'Simplified_AdEMAMix': {'title': 'Simplified AdEMAMix', 'tooltip': "Enables a simplified, single-EMA variant of AdEMAMix. Instead of blending two moving averages (fast and slow momentum), this version combines the raw current gradient (controlled by 'Grad α') directly with a single theory-based momentum. This makes the optimizer highly responsive to recent gradient information, which can accelerate training in all batch size scenarios when tuned correctly.", 'type': 'bool'},
            'alpha_grad': {'title': 'Grad α', 'tooltip': 'Controls the mixing coefficient between raw gradients and momentum gradients in Simplified AdEMAMix. Higher values (e.g., 10-100) emphasize recent gradients, suitable for small batch sizes to reduce noise. Lower values (e.g., 0-1) emphasize historical gradients, suitable for large batch sizes for stability. Setting to 0 uses only momentum gradients without raw gradient contribution.', 'type': 'float'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
            'k_warmup_steps': {'title': 'K-β Warmup Steps ', 'tooltip': 'When using Kourkoutas Beta, the number of initial training steps during which the dynamic β₂ logic is held off. In this period, β₂ is set to its fixed value to allow for initial training stability before the adaptive mechanism activates.', 'type': 'int'},
        }
        # @formatter:on

        adam_params = self.adam_params_def

        for index, key in enumerate(adam_params.keys()):
            if key not in KEY_DETAIL_MAP:
                continue

            arg_info = KEY_DETAIL_MAP[key]

            title = arg_info['title']
            tooltip = arg_info['tooltip']
            param_type = arg_info['type']

            row = index // 2
            col = 3 * (index % 2)

            components.label(master, row, col, title, tooltip=tooltip)

            if param_type != 'bool':
                components.entry(master, row, col + 1, self.adam_ui_state, key)
            else:
                components.switch(master, row, col + 1, self.adam_ui_state, key)
