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
            'orthogonal_gradient': {'title': 'OrthoGrad', 'tooltip': 'Reduces overfitting by removing the gradient component parallel to the weight, thus improving generalization. This has two modes: 1. flattened: Standard vectorized OrthoGrad. Fastest, but loses the structural properties of matrices. 2. iterative: Matrix-wise OrthoGrad, preserves structure by iteratively projecting rows and columns.', 'type': 'OrthoGrad'},
            'use_atan2': {'title': 'Atan2 Scaling', 'tooltip': 'A robust replacement for eps, which also incorporates gradient clipping, bounding and stabilizing the optimizer updates.', 'type': 'bool'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
            'nesterov_coef': {'title': 'Nesterov Coef', 'tooltip': 'Controls the mixing coefficient between momentum gradients and raw gradients in Nesterov momentum. For a factor of 0.8, the final update will be 80% of the momentum gradients and 20% raw gradient. Leaving it unset toggles the standard Nestrov behavior (where nesterov_coef = beta1 or momentum). Setting it to 0 cancels momentum contribution.', 'type': 'float'},
            'factored_2nd': {'title': 'Factored 2nd', 'tooltip': 'Whether to keep the first moment uncompressed (dense), while only factorizing the second moment. This makes the optimizer highly robust to a wide range of LRs, mimicking high-order optimization.', 'type': 'bool'},
            'fisher_wd': {'title': 'Fisher Weight Decay', 'tooltip': 'Applies adaptive, scale-invariant weight-decay regularization based on the Fisher Information Matrix (approximated by Adam\'s second moment). It reduces penalty for "important" high-curvature weights while accelerating decay for "useless" weights in flat regions. Leading to improved convergence and better final performance.', 'type': 'bool'},
            'state_precision': {'title': 'state_precision', 'tooltip': """The quantization format used to store the optimizer states to save VRAM. Options include: 'auto': Stores the states in the original parameter's precision. 'factored': Enables a memory-efficient mode by applying fast low-rank factorization to the optimizers states. It combines factorization for magnitudes with 1-bit compression for signs, drastically reducing VRAM usage and allowing for larger models or batch sizes. 'fp32': Uses full FP32. 'bf16_sr': Uses BF16 with stochastic rounding for a balance of precision and memory. 'int8_sr': Uses 8-bit block-wise quantization with stochastic rounding.""", 'type': 'StatePrecision'},
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
            elif param_type == 'StatePrecision':
                components.options(master, row, col + 1, ["auto", "factored", "fp32", "bf16_sr", "int8_sr"], self.adam_ui_state, key)
            elif param_type == 'OrthoGrad':
                components.options(master, row, col + 1, ["disabled", "flattened", "iterative"], self.adam_ui_state, key)
            else:
                components.switch(master, row, col + 1, self.adam_ui_state, key)
