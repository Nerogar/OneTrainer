import math
from typing import Callable


def lr_lambda_warmup(warmup_steps: int, lr_lambda: Callable[[int], float]):
    def warmup(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        else:
            return lr_lambda(current_step - warmup_steps)

    return warmup


def lr_lambda_constant():
    def lr_lambda(current_step: int):
        return 1

    return lr_lambda


def lr_lambda_linear(
        warmup_steps: int,
        max_epochs: int,
        approximate_epoch_length: int,
):
    scheduler_steps = max_epochs * approximate_epoch_length - warmup_steps

    def lr_lambda(current_step: int):
        return max(0.0, float(scheduler_steps - current_step) / float(scheduler_steps))

    return lr_lambda


def lr_lambda_cosine(
        warmup_steps: int,
        max_epochs: int,
        approximate_epoch_length: int,
):
    scheduler_steps = max_epochs * approximate_epoch_length - warmup_steps

    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(progress * math.pi)
        return max(0.0, 0.5 * (1.0 + schedule))

    return lr_lambda


def lr_lambda_cosine_with_restarts(
        warmup_steps: int,
        num_cycles: float,
        max_epochs: int,
        approximate_epoch_length: int,
):
    scheduler_steps = max_epochs * approximate_epoch_length - warmup_steps

    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(progress * 2.0 * math.pi * num_cycles)
        return max(0.0, 0.5 * (1.0 + schedule))

    return lr_lambda


def lr_lambda_cosine_with_hard_restarts(
        warmup_steps: int,
        num_cycles: float,
        max_epochs: int,
        approximate_epoch_length: int,
):
    scheduler_steps = max_epochs * approximate_epoch_length - warmup_steps

    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(((progress * num_cycles) % 1.0) * math.pi)
        return max(0.0, 0.5 * (1.0 + schedule))

    return lr_lambda
