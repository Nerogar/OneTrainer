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
        scheduler_steps: int,
):
    def lr_lambda(current_step: int):
        return max(0.0, float(scheduler_steps - current_step) / float(scheduler_steps))

    return lr_lambda


def lr_lambda_cosine(
        scheduler_steps: int,
):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(progress * math.pi)
        return max(0.0, 0.5 * (1.0 + schedule))

    return lr_lambda


def lr_lambda_cosine_with_restarts(
        scheduler_steps: int,
        num_cycles: float,
):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(progress * 2.0 * math.pi * num_cycles)
        return max(0.0, 0.5 * (1.0 + schedule))

    return lr_lambda


def lr_lambda_cosine_with_hard_restarts(
        scheduler_steps: int,
        num_cycles: float,
):
    def lr_lambda(current_step: int):
        progress = float(current_step) / float(scheduler_steps)
        schedule = math.cos(((progress * num_cycles) % 1.0) * math.pi)
        return max(0.0, 0.5 * (1.0 + schedule))

    return lr_lambda


def lr_lambda_rex(
        scheduler_steps: int,
):
    def lr_lambda(current_step: int):
        # https://arxiv.org/abs/2107.04197
        max_lr = 1
        min_lr = 0
        d = 0.9

        if current_step < scheduler_steps:
            progress = (current_step / scheduler_steps)
            div = (1 - d) + (d * (1 - progress))
            return min_lr + (max_lr - min_lr) * ((1 - progress) / div)
        else:
            return min_lr

    return lr_lambda
