from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.Optimizer import Optimizer


def change_optimizer(train_config: TrainConfig) -> TrainOptimizerConfig:
    optimizer = train_config.optimizer.optimizer

    optimizer_config = TrainOptimizerConfig.default_values()
    optimizer_config.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[optimizer])
    optimizer_config.optimizer = optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        saved_optimizer_config = train_config.optimizer_defaults[str(optimizer)]
        optimizer_config.from_dict(saved_optimizer_config.to_dict())

    return optimizer_config


def load_optimizer_defaults(train_config: TrainConfig) -> TrainOptimizerConfig:
    optimizer = train_config.optimizer.optimizer

    optimizer_config = TrainOptimizerConfig.default_values()
    optimizer_config.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[optimizer])
    optimizer_config.optimizer = optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        train_config.optimizer_defaults.pop(str(optimizer))

    return optimizer_config


def update_optimizer_config(train_config: TrainConfig):
    optimizer = train_config.optimizer.optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        saved_optimizer_config = train_config.optimizer_defaults[str(optimizer)]
        saved_optimizer_config.from_dict(train_config.optimizer.to_dict())
    else:
        optimizer_donfig = TrainOptimizerConfig.default_values()
        optimizer_donfig.from_dict(train_config.optimizer.to_dict())
        train_config.optimizer_defaults[str(optimizer)] = optimizer_donfig


# Optimizer Key map with defaults
OPTIMIZER_DEFAULT_PARAMETERS = {
    Optimizer.ADAFACTOR: {
        "eps": 1e-30,
        "eps2": 1e-3,
        "clip_threshold": 1.0,
        "decay_rate": -0.8,
        "beta1": None,
        "weight_decay": 0.0,
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
        "stochastic_rounding": True,
        "fused_back_pass": False,
    },
    Optimizer.ADAGRAD: {
        "lr_decay": 0,
        "weight_decay": 0,
        "initial_accumulator_value": 0,
        "eps": 1e-10,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.ADAGRAD_8BIT: {
        "lr_decay": 0,
        "weight_decay": 0,
        "initial_accumulator_value": 0,
        "eps": 1e-10,
        "optim_bits": 8,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.ADAM_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.ADAMW_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.LAMB: {
        "bias_correction": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "adam_w_mode": True,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": False,
        "max_unorm": 1.0,
    },
    Optimizer.LAMB_8BIT: {
        "bias_correction": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "adam_w_mode": True,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": False,
        "max_unorm": 1.0,
    },
    Optimizer.LARS: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "max_unorm": 0.02,
    },
    Optimizer.LARS_8BIT: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "max_unorm": 0.02,
    },
    Optimizer.LION_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.RMSPROP: {
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.RMSPROP_8BIT: {
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.SGD_8BIT: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.PRODIGY: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": None,
        "eps": 1e-8,
        "weight_decay": 0,
        "decouple": True,
        "use_bias_correction": False,
        "safeguard_warmup": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_ADA_GRAD: {
        "momentum": 0,
        "log_every": 0,
        "weight_decay": 0.0,
        "eps": 0.0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
    },
    Optimizer.DADAPT_ADAN: {
        "beta1": 0.98,
        "beta2": 0.92,
        "beta3": 0.99,
        "eps": 1e-8,
        "weight_decay": 0.02,
        "no_prox": False,
        "log_every": 0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
    },
    Optimizer.DADAPT_ADAM: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "log_every": 0,
        "decouple": False,
        "use_bias_correction": False,
        "d0": 1e-6,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_SGD: {
        "momentum": 0.0,
        "weight_decay": 0,
        "log_every": 0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_LION: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "log_every": 0,
        "d0": 1e-6,
        "fsdp_in_use": False,
    },
    Optimizer.ADAM: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "stochastic_rounding": False,
    },
    Optimizer.ADAMW: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "stochastic_rounding": False,
    },
    Optimizer.SGD: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "foreach": False,
        "maximize": False,
        "differentiable": False,
    },
    Optimizer.LION: {
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 0.0,
        "use_triton": False,
    },
}
