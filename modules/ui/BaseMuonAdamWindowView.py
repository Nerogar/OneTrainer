

class BaseMuonAdamWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, master, controller, ui_state):
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
            'use_AdEMAMix': {'title': 'AdEMAMix EMA', 'tooltip': 'Adds a second, slow-moving EMA, which is combined with the primary momentum to stabilize updates, and accelerate the training.', 'type': 'bool'},
            'beta3_ema': {'title': 'Beta3 EMA', 'tooltip': 'Coefficient for slow-moving EMA of AdEMAMix.', 'type': 'float'},
            'Simplified_AdEMAMix': {'title': 'Simplified AdEMAMix', 'tooltip': "Enables a simplified, single-EMA variant of AdEMAMix. Instead of blending two moving averages (fast and slow momentum), this version combines the raw current gradient (controlled by 'Grad α') directly with a single theory-based momentum. This makes the optimizer highly responsive to recent gradient information, which can accelerate training in all batch size scenarios when tuned correctly.", 'type': 'bool'},
            'alpha_grad': {'title': 'Grad α', 'tooltip': 'Controls the mixing coefficient between raw gradients and momentum gradients in Simplified AdEMAMix. Higher values (e.g., 10-100) emphasize recent gradients, suitable for small batch sizes to reduce noise. Lower values (e.g., 0-1) emphasize historical gradients, suitable for large batch sizes for stability. Setting to 0 uses only momentum gradients without raw gradient contribution.', 'type': 'float'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
        }
        # @formatter:on

        adam_params = controller.get_adam_params_def()

        for index, key in enumerate(adam_params.keys()):
            if key not in KEY_DETAIL_MAP:
                continue

            arg_info = KEY_DETAIL_MAP[key]

            title = arg_info['title']
            tooltip = arg_info['tooltip']
            param_type = arg_info['type']

            row = index // 2
            col = 3 * (index % 2)

            self.components.label(master, row, col, title, tooltip=tooltip)

            if param_type != 'bool':
                self.components.entry(master, row, col + 1, ui_state, key)
            else:
                self.components.switch(master, row, col + 1, ui_state, key)
