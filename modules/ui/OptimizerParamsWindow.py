import contextlib
from tkinter import TclError

from modules.ui.MuonAdamWindow import MUON_AUX_ADAM_DEFAULTS, MuonAdamWindow
from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import (
    OPTIMIZER_DEFAULT_PARAMETERS,
    change_optimizer,
    load_optimizer_defaults,
    update_optimizer_config,
)
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class OptimizerParamsWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            train_config: TrainConfig,
            ui_state,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state
        self.optimizer_ui_state = ui_state.get_var("optimizer")
        self.protocol("WM_DELETE_WINDOW", self.on_window_close)
        self.muon_adam_button = None

        self.title("Optimizer Settings")
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

        components.button(self, 1, 0, "ok", command=self.on_window_close)
        self.main_frame(self.frame)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))


    def main_frame(self, master):
        # Optimizer
        components.label(master, 0, 0, "Optimizer",
                         tooltip="The type of optimizer")

        # Create the optimizer dropdown menu and set the command
        components.options(master, 0, 1, [str(x) for x in list(Optimizer)], self.optimizer_ui_state, "optimizer",
                           command=self.on_optimizer_change)

        # Defaults Button
        components.label(master, 0, 3, "Optimizer Defaults",
                         tooltip="Load default settings for the selected optimizer")
        components.button(self.frame, 0, 4, "Load Defaults", self.load_defaults,
                          tooltip="Load default settings for the selected optimizer")

        self.create_dynamic_ui(master)

    def clear_dynamic_ui(self, master):
        with contextlib.suppress(TclError):
            for widget in master.winfo_children():
                grid_info = widget.grid_info()
                if int(grid_info["row"]) >= 1:
                    widget.destroy()

    def create_dynamic_ui(
            self,
            master,
    ):

        # Lookup for the title and tooltip for a key
        # @formatter:off
        KEY_DETAIL_MAP = {
            'adam_w_mode': {'title': 'Adam W Mode', 'tooltip': 'Whether to use weight decay correction for Adam optimizer.', 'type': 'bool'},
            'alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'amsgrad': {'title': 'AMSGrad', 'tooltip': 'Whether to use the AMSGrad variant for Adam.', 'type': 'bool'},
            'beta1': {'title': 'Beta1', 'tooltip': 'optimizer_momentum term.', 'type': 'float'},
            'beta2': {'title': 'Beta2', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'float'},
            'beta3': {'title': 'Beta3', 'tooltip': 'Coefficient for computing the Prodigy stepsize.', 'type': 'float'},
            'bias_correction': {'title': 'Bias Correction', 'tooltip': 'Whether to use bias correction in optimization algorithms like Adam.', 'type': 'bool'},
            'block_wise': {'title': 'Block Wise', 'tooltip': 'Whether to perform block-wise model update.', 'type': 'bool'},
            'capturable': {'title': 'Capturable', 'tooltip': 'Whether some property of the optimizer can be captured.', 'type': 'bool'},
            'centered': {'title': 'Centered', 'tooltip': 'Whether to center the gradient before scaling. Great for stabilizing the training process.', 'type': 'bool'},
            'clip_threshold': {'title': 'Clip Threshold', 'tooltip': 'Clipping value for gradients.', 'type': 'float'},
            'd0': {'title': 'Initial D', 'tooltip': 'Initial D estimate for D-adaptation.', 'type': 'float'},
            'd_coef': {'title': 'D Coefficient', 'tooltip': 'Coefficient in the expression for the estimate of d.', 'type': 'float'},
            'dampening': {'title': 'Dampening', 'tooltip': 'Dampening for optimizer_momentum.', 'type': 'float'},
            'decay_rate': {'title': 'Decay Rate', 'tooltip': 'Rate of decay for moment estimation.', 'type': 'float'},
            'decouple': {'title': 'Decouple', 'tooltip': 'Use AdamW style optimizer_decoupled weight decay.', 'type': 'bool'},
            'differentiable': {'title': 'Differentiable', 'tooltip': 'Whether the optimization function is optimizer_differentiable.', 'type': 'bool'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'eps2': {'title': 'EPS 2', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'foreach': {'title': 'ForEach', 'tooltip': 'Whether to use a foreach implementation if available. This implementation is usually faster.', 'type': 'bool'},
            'fsdp_in_use': {'title': 'FSDP in Use', 'tooltip': 'Flag for using sharded parameters.', 'type': 'bool'},
            'fused': {'title': 'Fused', 'tooltip': 'Whether to use a fused implementation if available. This implementation is usually faster and requires less memory.', 'type': 'bool'},
            'fused_back_pass': {'title': 'Fused Back Pass', 'tooltip': 'Whether to fuse the back propagation pass with the optimizer step. This reduces VRAM usage, but is not compatible with gradient accumulation.', 'type': 'bool'},
            'growth_rate': {'title': 'Growth Rate', 'tooltip': 'Limit for D estimate growth rate.', 'type': 'float'},
            'initial_accumulator_value': {'title': 'Initial Accumulator Value', 'tooltip': 'Initial value for Adagrad optimizer.', 'type': 'float'},
            'initial_accumulator': {'title': 'Initial Accumulator', 'tooltip': 'Sets the starting value for both moment estimates to ensure numerical stability and balanced adaptive updates early in training.', 'type': 'float'},
            'is_paged': {'title': 'Is Paged', 'tooltip': 'Whether the optimizer\'s internal state should be paged to CPU.', 'type': 'bool'},
            'log_every': {'title': 'Log Every', 'tooltip': 'Intervals at which logging should occur.', 'type': 'int'},
            'lr_decay': {'title': 'LR Decay', 'tooltip': 'Rate at which learning rate decreases.', 'type': 'float'},
            'max_unorm': {'title': 'Max Unorm', 'tooltip': 'Maximum value for gradient clipping by norms.', 'type': 'float'},
            'maximize': {'title': 'Maximize', 'tooltip': 'Whether to optimizer_maximize the optimization function.', 'type': 'bool'},
            'min_8bit_size': {'title': 'Min 8bit Size', 'tooltip': 'Minimum tensor size for 8-bit quantization.', 'type': 'int'},
            'quant_block_size': {'title': 'Quant Block Size', 'tooltip': 'Size of a block of normalized 8-bit quantization data. Larger values increase memory efficiency at the cost of data precision.', 'type': 'int'},
            'momentum': {'title': 'optimizer_momentum', 'tooltip': 'Factor to accelerate SGD in relevant direction.', 'type': 'float'},
            'nesterov': {'title': 'Nesterov', 'tooltip': 'Whether to enable Nesterov optimizer_momentum.', 'type': 'bool'},
            'no_prox': {'title': 'No Prox', 'tooltip': 'Whether to use proximity updates or not.', 'type': 'bool'},
            'optim_bits': {'title': 'Optim Bits', 'tooltip': 'Number of bits used for optimization.', 'type': 'int'},
            'percentile_clipping': {'title': 'Percentile Clipping', 'tooltip': 'Gradient clipping based on percentile values.', 'type': 'int'},
            'relative_step': {'title': 'Relative Step', 'tooltip': 'Whether to use a relative step size.', 'type': 'bool'},
            'safeguard_warmup': {'title': 'Safeguard Warmup', 'tooltip': 'Avoid issues during warm-up stage.', 'type': 'bool'},
            'scale_parameter': {'title': 'Scale Parameter', 'tooltip': 'Whether to scale the parameter or not.', 'type': 'bool'},
            'stochastic_rounding': {'title': 'Stochastic Rounding', 'tooltip': 'Stochastic rounding for weight updates. Improves quality when using bfloat16 weights.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'use_triton': {'title': 'Use Triton', 'tooltip': 'Whether Triton optimization should be used.', 'type': 'bool'},
            'warmup_init': {'title': 'Warmup Initialization', 'tooltip': 'Whether to warm-up the optimizer initialization.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
            'weight_lr_power': {'title': 'Weight LR Power', 'tooltip': 'During warmup, the weights in the average will be equal to lr raised to this power. Set to 0 for no weighting.', 'type': 'float'},
            'decoupled_decay': {'title': 'Decoupled Decay', 'tooltip': 'If set as True, then the optimizer uses decoupled weight decay as in AdamW.', 'type': 'bool'},
            'fixed_decay': {'title': 'Fixed Decay', 'tooltip': '(When Decoupled Decay is True:) Applies fixed weight decay when True; scales decay with learning rate when False.', 'type': 'bool'},
            'rectify': {'title': 'Rectify', 'tooltip': 'Perform the rectified update similar to RAdam.', 'type': 'bool'},
            'degenerated_to_sgd': {'title': 'Degenerated to SGD', 'tooltip': 'Performs SGD update when gradient variance is high.', 'type': 'bool'},
            'k': {'title': 'K', 'tooltip': 'Number of vector projected per iteration.', 'type': 'int'},
            'xi': {'title': 'Xi', 'tooltip': 'Term used in vector projections to avoid division by zero.', 'type': 'float'},
            'n_sma_threshold': {'title': 'N SMA Threshold', 'tooltip': 'Number of SMA threshold.', 'type': 'int'},
            'ams_bound': {'title': 'AMS Bound', 'tooltip': 'Whether to use the AMSBound variant.', 'type': 'bool'},
            'r': {'title': 'R', 'tooltip': 'EMA factor.', 'type': 'float'},
            'adanorm': {'title': 'AdaNorm', 'tooltip': 'Whether to use the AdaNorm variant', 'type': 'bool'},
            'adam_debias': {'title': 'Adam Debias', 'tooltip': 'Only correct the denominator to avoid inflating step sizes early in training.', 'type': 'bool'},
            'slice_p': {'title': 'Slice parameters', 'tooltip': 'Reduce memory usage by calculating LR adaptation statistics on only every pth entry of each tensor. For values greater than 1 this is an approximation to standard Prodigy. Values ~11 are reasonable.', 'type': 'int'},
            'cautious': {'title': 'Cautious', 'tooltip': 'Whether to use the Cautious variant', 'type': 'bool'},
            'weight_decay_by_lr': {'title': 'weight_decay_by_lr', 'tooltip': 'Automatically adjust weight decay based on lr', 'type': 'bool'},
            'prodigy_steps': {'title': 'prodigy_steps', 'tooltip': 'Turn off Prodigy after N steps', 'type': 'int'},
            'use_speed': {'title': 'use_speed', 'tooltip': 'use_speed method', 'type': 'bool'},
            'split_groups': {'title': 'split_groups', 'tooltip': 'Use split groups when training multiple params(uNet,TE..)', 'type': 'bool'},
            'split_groups_mean': {'title': 'split_groups_mean', 'tooltip': 'Use mean for split groups', 'type': 'bool'},
            'factored': {'title': 'factored', 'tooltip': 'Use factored', 'type': 'bool'},
            'factored_fp32': {'title': 'factored_fp32', 'tooltip': 'Use factored_fp32', 'type': 'bool'},
            'use_stableadamw': {'title': 'use_stableadamw', 'tooltip': 'Use use_stableadamw for gradient scaling', 'type': 'bool'},
            'use_cautious': {'title': 'use_cautious', 'tooltip': 'Use cautious method', 'type': 'bool'},
            'use_grams': {'title': 'use_grams', 'tooltip': 'Use grams method', 'type': 'bool'},
            'use_adopt': {'title': 'use_adopt', 'tooltip': 'Use adopt method', 'type': 'bool'},
            'd_limiter': {'title': 'd_limiter', 'tooltip': 'Prevent over-estimated LRs when gradients and EMA are still stabilizing', 'type': 'bool'},
            'use_schedulefree': {'title': 'use_schedulefree', 'tooltip': 'Use Schedulefree method', 'type': 'bool'},
            'use_orthograd': {'title': 'use_orthograd', 'tooltip': 'Use orthograd method', 'type': 'bool'},
            'nnmf_factor': {'title': 'Factored Optimizer', 'tooltip': 'Enables a memory-efficient mode by applying fast low-rank factorization to the optimizers states. It combines factorization for magnitudes with 1-bit compression for signs, drastically reducing VRAM usage and allowing for larger models or batch sizes. This is an approximation which may slightly alter training dynamics.', 'type': 'bool'},
            'orthogonal_gradient': {'title': 'OrthoGrad', 'tooltip': 'Reduces overfitting by removing the gradient component parallel to the weight, thus improving generalization.', 'type': 'bool'},
            'use_atan2': {'title': 'Atan2 Scaling', 'tooltip': 'A robust replacement for eps, which also incorporates gradient clipping, bounding and stabilizing the optimizer updates.', 'type': 'bool'},
            'cautious_mask': {'title': 'Cautious Variant', 'tooltip': 'Applies a mask to dampen or zero-out momentum components that disagree with the current gradients direction.', 'type': 'bool'},
            'grams_moment': {'title': 'GRAMS Variant', 'tooltip': 'Aligns the momentum direction with the current gradient direction while preserving its accumulated magnitude.', 'type': 'bool'},
            'use_AdEMAMix': {'title': 'AdEMAMix EMA', 'tooltip': 'Adds a second, slow-moving EMA, which is combined with the primary momentum to stabilize updates, and accelerate the training.', 'type': 'bool'},
            'beta3_ema': {'title': 'Beta3 EMA', 'tooltip': 'Coefficient for slow-moving EMA of AdEMAMix.', 'type': 'float'},
            'beta1_warmup': {'title': 'Beta1 Warmup Steps', 'tooltip': 'Number of warmup steps to gradually increase beta1 from Minimum Beta1 Value to its final value. During warmup, beta1 increases linearly. leave it empty to disable warmup and use constant beta1.', 'type': 'int'},
            'min_beta1': {'title': 'Minimum Beta1', 'tooltip': 'Starting beta1 value for warmup scheduling. Used only when beta1 warmup is enabled. Lower values allow faster initial adaptation, while higher values provide more smoothing. The final beta1 value is specified in the beta1 parameter.', 'type': 'float'},
            'Simplified_AdEMAMix': {'title': 'Simplified AdEMAMix', 'tooltip': "Enables a simplified, single-EMA variant of AdEMAMix. Instead of blending two moving averages (fast and slow momentum), this version combines the raw current gradient (controlled by 'Grad α') directly with a single theory-based momentum. This makes the optimizer highly responsive to recent gradient information, which can accelerate training in all batch size scenarios when tuned correctly.", 'type': 'bool'},
            'alpha_grad': {'title': 'Grad α', 'tooltip': 'Controls the mixing coefficient between raw gradients and momentum gradients in Simplified AdEMAMix. Higher values (e.g., 10-100) emphasize recent gradients, suitable for small batch sizes to reduce noise. Lower values (e.g., 0-1) emphasize historical gradients, suitable for large batch sizes for stability. Setting to 0 uses only momentum gradients without raw gradient contribution.', 'type': 'float'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
            'k_warmup_steps': {'title': 'K-β Warmup Steps ', 'tooltip': 'When using Kourkoutas Beta, the number of initial training steps during which the dynamic β₂ logic is held off. In this period, β₂ is set to its fixed value to allow for initial training stability before the adaptive mechanism activates.', 'type': 'int'},
            'schedulefree_c': {'title': 'Schedule free averaging strength', 'tooltip': 'Larger values = more responsive (shorter averaging window); smaller values = smoother (longer window). Set to 0 to disable and use the original Schedule-Free rule. Short small batches (≈6-12); long/large-batch (≈50-200).', 'type': 'float'},
            'ns_steps': {'title': 'Newton-Schulz Iterations', 'tooltip': 'Controls the number of iterations for update orthogonalization. Higher values improve the updates quality but make each step slower. Lower values are faster per step but may be less effective.', 'type': 'int'},
            'MuonWithAuxAdam': {'title': 'MuonWithAuxAdam', 'tooltip': 'Whether to use the standard way of Muon. Non-hidden layers fallback to ADAMW, and MUON takes the rest. Note: The auxiliary Adam (ADAMW) is typically only relevant for training "full" LoRA (LoRA for all layers) or full finetune and is irrelevant for most common LoRA use cases.', 'type': 'bool'},
            'muon_hidden_layers': {'title': 'Hidden Layers', 'tooltip': 'Comma-separated list of hidden layers to train using Muon. Regular expressions (if toggled) are supported. Any model layer with a matching name will be trained using Muon. If None is provided it will default to using automatic way of finding hidden layers.', 'type': 'str'},
            'muon_adam_regex': {'title': 'Use Regex', 'tooltip': 'Whether to use regular expressions for hidden layers.', 'type': 'bool'},
            'muon_adam_lr': {'title': 'Auxiliary Adam LR', 'tooltip': 'Learning rate for the auxiliary AdamW optimizer. If empty, it will use the main learning rate.', 'type': 'float'},
            'muon_te1_adam_lr': {'title': 'AuxAdam TE1 LR', 'tooltip': 'Learning rate for the auxiliary AdamW optimizer for the first text encoder. If empty, it will use the Auxiliary Adam LR.', 'type': 'float'},
            'muon_te2_adam_lr': {'title': 'AuxAdam TE2 LR', 'tooltip': 'Learning rate for the auxiliary AdamW optimizer for the second text encoder. If empty, it will use the Auxiliary Adam LR.', 'type': 'float'},
            'rms_rescaling': {'title': 'RMS Rescaling', 'tooltip': 'Muon already scales its updates to approximate and use the same learning rate (LR) as Adam. This option integrates a more accurate method to match the Adam LR, but it is slower.', 'type': 'bool'},
            'normuon_variant': {'title': 'NorMuon Variant', 'tooltip': 'Enables the NorMuon optimizer variant, which combines Muon orthogonalization with per-neuron adaptive learning rates for better convergence and balanced parameter updates. Costs only one scalar state buffer per parameter group, size few KBs, maintaining high memory efficiency.', 'type': 'bool'},
            'beta2_normuon': {'title': 'NorMuon Beta2', 'tooltip': 'Exponential decay rate for the neuron-wise second-moment estimator in NorMuon (analogous to Adams beta2). Controls how past squared updates influence current normalization.', 'type': 'float'},
            'normuon_eps': {'title': 'NorMuon EPS', 'tooltip': 'Epsilon for NorMuon normalization stability.', 'type': 'float'},
            'low_rank_ortho': {'title': 'Low-rank Orthogonalization', 'tooltip': 'Use low-rank orthogonalization to accelerate Muon by orthogonalizing only in a low-dimensional subspace, improving speed and noise robustness.', 'type': 'bool'},
            'ortho_rank': {'title': 'Ortho Rank', 'tooltip': 'Target rank for low-rank orthogonalization. Controls the dimensionality of the subspace used for efficient and noise-robust orthogonalization.', 'type': 'int'},
            'accelerated_ns': {'title': 'Accelerated Newton-Schulz', 'tooltip': 'Applies an enhanced Newton-Schulz variant that replaces heuristic coefficients with optimal coefficients derived at each step. This improves performance and convergence by reducing the number of required operations.', 'type': 'bool'},
            'cautious_wd': {'title': 'Cautious Weight Decay', 'tooltip': 'Applies weight decay only to parameter coordinates whose signs align with the optimizer update direction. This preserves the original optimization objective while still benefiting from regularization effects, leading to improved convergence and better final performance.', 'type': 'bool'},
            'approx_mars': {'title': 'Approx MARS-M', 'tooltip': 'Enables Approximated MARS-M, a variance reduction technique. It uses the previous step\'s gradient to correct the current update, leading to lower losses and improved convergence stability. This requires additional state to store the previous gradient.', 'type': 'bool'},
            'kappa_p': {'title': 'Lion-K P-value', 'tooltip': 'Controls the Lp-norm geometry for the Lion update. 1.0 = Standard Lion (Sign update, coordinate-wise), best for Transformers. 2.0 = Spherical Lion (Normalized update, rotational invariant), best for Conv2d layers (in unet models). Values between 1.0 and 2.0 interpolate behavior between the two.', 'type': 'float'},
            'auto_kappa_p': {'title': 'Auto Lion-K', 'tooltip': 'Automatically determines the optimal P-value based on layer dimensions. Uses p=2.0 (Spherical) for 4D (Conv) tensors for stability and rotational invariance, and p=1.0 (Sign) for 2D (Linear) tensors for sparsity. Overrides the manual P-value. Recommend for unet models.', 'type': 'bool'},
        }
        # @formatter:on

        if not self.winfo_exists():  # check if this window isn't open
            return

        selected_optimizer = self.train_config.optimizer.optimizer

        # Extract the keys for the selected optimizer
        for index, key in enumerate(OPTIMIZER_DEFAULT_PARAMETERS[selected_optimizer].keys()):
            if key not in KEY_DETAIL_MAP:
                continue
            arg_info = KEY_DETAIL_MAP[key]

            title = arg_info['title']
            tooltip = arg_info['tooltip']
            type = arg_info['type']

            row = (index // 2) + 1
            col = 3 * (index % 2)

            components.label(master, row, col, title, tooltip=tooltip)

            if key == 'MuonWithAuxAdam':
                frame = ctk.CTkFrame(master, fg_color="transparent")
                frame.grid(row=row, column=col + 1, columnspan=2, sticky="ew", padx=0, pady=0)
                frame.grid_columnconfigure(0, weight=0)
                frame.grid_columnconfigure(1, weight=0)

                components.switch(frame, 0, 0, self.optimizer_ui_state, key, command=self.update_user_pref)

                self.muon_adam_button = components.button(
                    frame, 0, 1, "...", self.open_muon_adam_window,
                    tooltip="Configure the auxiliary AdamW_adv optimizer",
                    width=20, padx=5                )
                self.toggle_muon_adam_button()
            elif type != 'bool':
                components.entry(master, row, col + 1, self.optimizer_ui_state, key,
                                 command=self.update_user_pref)
            else:
                components.switch(master, row, col + 1, self.optimizer_ui_state, key,
                                  command=self.update_user_pref)

    def update_user_pref(self, *args):
        update_optimizer_config(self.train_config)
        self.toggle_muon_adam_button()

    def on_optimizer_change(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

        self.clear_dynamic_ui(self.frame)
        self.create_dynamic_ui(self.frame)

    def load_defaults(self, *args):
        optimizer_config = load_optimizer_defaults(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

    def on_window_close(self):
        self.destroy()

    def toggle_muon_adam_button(self):
        if self.muon_adam_button and self.muon_adam_button.winfo_exists():
            muon_with_adam = self.optimizer_ui_state.get_var("MuonWithAuxAdam").get()
            self.muon_adam_button.configure(state="normal" if muon_with_adam else "disabled")

    def open_muon_adam_window(self):
        current_optimizer = self.train_config.optimizer.optimizer

        adam_config = TrainOptimizerConfig.default_values()
        current_state = self.train_config.optimizer.muon_adam_config

        if current_optimizer == Optimizer.MUON:
            defaults = MUON_AUX_ADAM_DEFAULTS
        else:
            defaults = OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.ADAMW_ADV]

        if current_state is None:
            adam_config.from_dict(defaults)
            if current_optimizer != Optimizer.MUON:
                adam_config.optimizer = Optimizer.ADAMW_ADV
        elif isinstance(current_state, dict):
            adam_config.from_dict(current_state)
        else:
            # Should not happen if TrainConfig defines it as dict, but for safety
            adam_config = current_state

        temp_adam_ui_state = UIState(self, adam_config)
        window = MuonAdamWindow(self, self.train_config, temp_adam_ui_state, current_optimizer)
        self.wait_window(window)

        self.train_config.optimizer.muon_adam_config = adam_config.to_dict()
