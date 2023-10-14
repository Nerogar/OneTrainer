from modules.util.enum.Optimizer import Optimizer
import json
import os

class UserPreferenceUtility:
    def __init__(self, file_path="training_user_settings/optimizer_prefs.json"):
        self.file_path = file_path
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.mkdir(directory)

    def load_preferences(self, optimizer_name):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                prefs = json.load(f)
                return prefs.get(optimizer_name, "Use_Default")
        return "Use_Default"

    def save_preference(self, optimizer_name, key, value):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                prefs = json.load(f)
        else:
            prefs = {}

        if optimizer_name not in prefs:
            prefs[optimizer_name] = {}

        prefs[optimizer_name][key] = value

        with open(self.file_path, 'w') as f:
            json.dump(prefs, f, indent=4)

    def remove_preference(self, optimizer_name):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                prefs = json.load(f)
        else:
            prefs = {}

        if optimizer_name in prefs:
            prefs.pop(optimizer_name)

        with open(self.file_path, 'w') as f:
            json.dump(prefs, f, indent=4)

# Optimizer Key map with defaults
OPTIMIZER_KEY_MAP = {
    "ADAFACTOR": {
        "optimizer_eps": 1e-30,
        "optimizer_eps2": 1e-3,
        "optimizer_clip_threshold": 1.0,
        "optimizer_decay_rate": -0.8,
        "optimizer_beta1": None,
        "optimizer_weight_decay": 0.0,
        "optimizer_scale_parameter": True,
        "optimizer_relative_step": True,
        "optimizer_warmup_init": False,
    },
    "ADAGRAD": {
        "optimizer_lr_decay": 0,
        "optimizer_weight_decay": 0,
        "optimizer_initial_accumulator_value": 0,
        "optimizer_eps": 1e-10,
        "optimizer_optim_bits": 32,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
    },
    "ADAGRAD_8BIT": {
        "optimizer_lr_decay": 0,
        "optimizer_weight_decay": 0,
        "optimizer_initial_accumulator_value": 0,
        "optimizer_eps": 1e-10,
        "optimizer_optim_bits": 8,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
    },
    "ADAM_8BIT": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_amsgrad": False,
        "optimizer_optim_bits": 32,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
        "optimizer_is_paged": False,
    },
    "ADAMW_8BIT": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 1e-2,
        "optimizer_amsgrad": False,
        "optimizer_optim_bits": 32,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
        "optimizer_is_paged": False,
    },
    "LAMB": {
        "optimizer_bias_correction": True,
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_amsgrad": False,
        "optimizer_adam_w_mode": True,
        "optimizer_optim_bits": 32,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": False,
        "optimizer_max_unorm": 1.0,
    },
    "LAMB_8BIT": {
        "optimizer_bias_correction": True,
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_amsgrad": False,
        "optimizer_adam_w_mode": True,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": False,
        "optimizer_max_unorm": 1.0,
    },
    "LARS": {
        "optimizer_momentum": 0,
        "optimizer_dampening": 0,
        "optimizer_weight_decay": 0,
        "optimizer_nesterov": False,
        "optimizer_optim_bits": 32,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_max_unorm": 0.02,
    },
    "LARS_8BIT": {
        "optimizer_momentum": 0,
        "optimizer_dampening": 0,
        "optimizer_weight_decay": 0,
        "optimizer_nesterov": False,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_max_unorm": 0.02,
    },
    "LION_8BIT": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_weight_decay": 0,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
        "optimizer_is_paged": False,
    },
    "RMSPROP": {
        "optimizer_alpha": 0.99,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_momentum": 0,
        "optimizer_centered": False,
        "optimizer_optim_bits": 32,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
    },
    "RMSPROP_8BIT": {
        "optimizer_alpha": 0.99,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_momentum": 0,
        "optimizer_centered": False,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
    },
    "SGD_8BIT": {
        "optimizer_momentum": 0,
        "optimizer_dampening": 0,
        "optimizer_weight_decay": 0,
        "optimizer_nesterov": False,
        "optimizer_min_8bit_size": 4096,
        "optimizer_percentile_clipping": 100,
        "optimizer_block_wise": True,
    },
    "PRODIGY": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_beta3": None,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_decouple": True,
        "optimizer_use_bias_correction": False,
        "optimizer_safeguard_warmup": False,
        "optimizer_d0": 1e-6,
        "optimizer_d_coef": 1.0,
        "optimizer_growth_rate": float('inf'),
        "optimizer_fsdp_in_use": False,
    },
    "DADAPT_ADA_GRAD": {
        "optimizer_momentum": 0,
        "optimizer_log_every": 0,
        "optimizer_weight_decay": 0.0,
        "optimizer_eps": 0.0,
        "optimizer_d0": 1e-6,
        "optimizer_growth_rate": float('inf'),
    },
    "DADAPT_ADAN": {
        "optimizer_beta1": 0.98,
        "optimizer_beta2": 0.92,
        "optimizer_beta3": 0.99,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0.02,
        "optimizer_no_prox": False,
        "optimizer_log_every": 0,
        "optimizer_d0": 1e-6,
        "optimizer_growth_rate": float('inf'),
    },
    "DADAPT_ADAM": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_log_every": 0,
        "optimizer_decouple": False,
        "optimizer_use_bias_correction": False,
        "optimizer_d0": 1e-6,
        "optimizer_growth_rate": float('inf'),
        "optimizer_fsdp_in_use": False,
    },
    "DADAPT_SGD": {
        "optimizer_momentum": 0.0,
        "optimizer_weight_decay": 0,
        "optimizer_log_every": 0,
        "optimizer_d0": 1e-6,
        "optimizer_growth_rate": float('inf'),
        "optimizer_fsdp_in_use": False,
    },
    "DADAPT_LION": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_weight_decay": 0.0,
        "optimizer_log_every": 0,
        "optimizer_d0": 1e-6,
        "optimizer_fsdp_in_use": False,
    },
    "ADAM": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0,
        "optimizer_amsgrad": False,
        "optimizer_foreach": False,
        "optimizer_maximize": False,
        "optimizer_capturable": False,
        "optimizer_differentiable": False,
        "optimizer_fused": True
    },
    "ADAMW": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 1e-2,
        "optimizer_amsgrad": False,
        "optimizer_foreach": False,
        "optimizer_maximize": False,
        "optimizer_capturable": False,
        "optimizer_differentiable": False,
        "optimizer_fused": True
    },
    "SGD": {
        "optimizer_momentum": 0,
        "optimizer_dampening": 0,
        "optimizer_weight_decay": 0,
        "optimizer_nesterov": False,
        "optimizer_foreach": False,
        "optimizer_maximize": False,
        "optimizer_differentiable": False
    },
    "LION": {
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.99,
        "optimizer_weight_decay": 0.0,
        "optimizer_use_triton": False
    },
}