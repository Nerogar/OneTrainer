

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
            'nesterov': {'title': 'Nesterov', 'tooltip': 'Enable Nesterov momentum for the auxiliary AdamW optimizer.', 'type': 'bool'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'stochastic_rounding': {'title': 'Stochastic Rounding', 'tooltip': 'Stochastic rounding for weight updates. Improves quality when using bfloat16 weights.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization to prevent overfitting.', 'type': 'float'},
            'use_orthograd': {'title': 'use_orthograd', 'tooltip': 'Use orthograd method', 'type': 'bool'},
            'orthogonal_gradient': {'title': 'OrthoGrad', 'tooltip': 'Reduces overfitting by removing the gradient component parallel to the weight, thus improving generalization. This has two modes: 1. flattened: Standard vectorized OrthoGrad. Fastest, but loses the structural properties of matrices. 2. iterative: Matrix-wise OrthoGrad, preserves structure by iteratively projecting rows and columns.', 'type': 'OrthoGrad'},
            'use_atan2': {'title': 'Atan2 Scaling', 'tooltip': 'A robust replacement for eps, which also incorporates gradient clipping, bounding and stabilizing the optimizer updates.', 'type': 'bool'},
            'kourkoutas_beta': {'title': 'Kourkoutas Beta', 'tooltip': 'Enables a layer-wise dynamic β₂ adaptation. This feature makes the optimizer more responsive to "spiky" gradients by lowering β₂ during periods of high variance, and more stable during calm periods by raising β₂ towards its maximum. It can significantly improve training stability and final loss.', 'type': 'bool'},
            'nesterov_coef': {'title': 'Nesterov Momentum Mix', 'tooltip': 'Only used when Nesterov is enabled. Higher values favor momentum over the current gradient. Leave blank to use Beta1 or Momentum.', 'type': 'float'},
            'factored_2nd': {'title': 'Compress Second-Moment State', 'tooltip': "Saves GPU memory by compressing the optimizer's running estimate of gradient size. Momentum is not factored by this option. Compression can change training results.", 'type': 'bool'},
            'fisher_wd': {'title': 'Fisher Weight Decay', 'tooltip': 'Adjusts weight decay separately for each weight using recent gradient history. Applies less decay where gradients are larger. Requires nonzero weight decay.', 'type': 'bool'},
            'state_precision': {'title': 'Optimizer State Storage', 'tooltip': "Controls GPU memory used for the optimizer's training history. Auto matches the weights' precision. FP32 uses more precision and memory. BF16, Int8, and Factored offer lower-memory alternatives.", 'type': 'StatePrecision'},
            'centered_wd': {'title': 'Centered Weight Decay', 'tooltip': 'Pulls trained weights toward their starting values. Higher values resist change more; 0 disables it. For LoRA/OFT adapter weights, use normal Weight Decay. This is meant for full finetuning.', 'type': 'float'},
            'centered_wd_mode': {'title': 'Reference Weight Storage', 'tooltip': 'Controls how the starting weights are stored for centered weight decay. Full keeps their original precision. Float8 and Int8 use less GPU memory; Int4 compresses them further.', 'type': 'CenteredWDMode'},
            'spectral_normalization': {'title': 'Spectral Scaling', 'tooltip': 'Adjusts update sizes for different layer sizes and adapter ranks. This is designed to reduce the need for learning-rate retuning, but you should still check your results.', 'type': 'bool'},
            'stochastic_sign': {'title': 'Stochastic Sign', 'tooltip': 'Uses pre-conditioned randomness so signed updates can reflect the strength of the training signal, rather than only its direction.', 'type': 'bool'},
            'orthogonal_sinkhorn': {'title': 'Orthogonal Sinkhorn', 'tooltip': 'Applies iterative row and column orthogonal projection to make the updates orthogonal to the current weight, leading to robust regularization and better generalization.', 'type': 'bool'},
            'sinkhorn_iterations': {'title': 'Sinkhorn Iterations', 'tooltip': 'Number of passes used to balance update sizes across rows and columns. More passes take longer. Default: 5.', 'type': 'int'},
            'normed_momentum': {'title': 'Normalize Before Momentum', 'tooltip': 'Balances gradient sizes before combining them with past gradients. This changes how momentum smooths updates.', 'type': 'bool'},
            'geometric_wd': {'title': 'Geometric Weight Decay', 'tooltip': 'Regularizes weights based on the geometric structure of the optimizer. Compatible with cautious weight decay.', 'type': 'bool'},
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

            if param_type == 'StatePrecision':
                self.components.options(master, row, col + 1, ["auto", "factored", "fp32", "bf16_sr", "int8_sr"], ui_state, key)
            elif param_type == 'OrthoGrad':
                self.components.options(master, row, col + 1, ["disabled", "flattened", "iterative"], ui_state, key)
            elif param_type == 'CenteredWDMode':
                self.components.options(master, row, col + 1, ["full", "float8", "int8", "int4"], ui_state, key)
            elif param_type != 'bool':
                self.components.entry(master, row, col + 1, ui_state, key)
            else:
                self.components.switch(master, row, col + 1, ui_state, key)
