"""Deterministic stratified validation timestep + noise assignment.

Validation loss is averaged across the full timestep distribution by giving each
sample a unique, persistent timestep derived from its index. Splitting [0, 1]
into N equal bins and jittering the position within each bin gives an even
spread at any N while remaining reproducible: identical configs and datasets
yield identical assignments.
"""

import numpy as np

# Distinct large constants keep the timestep stream and the noise stream
# independent. Changing these values changes everyone's validation losses, so
# they are baked in.
VAL_SEED = 0
TIMESTEP_TAG = 0xA5F13C7B
NOISE_TAG = 0x5E2D88C1


def stratified_unit_position(i: int, n: int, seed: int = VAL_SEED) -> float:
    if n <= 0:
        return 0.0
    jitter = float(np.random.default_rng([int(seed), TIMESTEP_TAG, int(i)]).random())
    return (int(i) + jitter) / int(n)


def validation_noise_seed(i: int, seed: int = VAL_SEED) -> int:
    return int(np.random.default_rng([int(seed), NOISE_TAG, int(i)]).integers(0, 2 ** 31))


def apply_timestep_shift_unit(pos: float, shift: float) -> float:
    if shift == 1.0:
        return float(pos)
    return float(shift) * float(pos) / ((float(shift) - 1.0) * float(pos) + 1.0)
