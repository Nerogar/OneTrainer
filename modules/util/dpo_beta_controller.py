import math


class DPOBetaController:
    """Batch-level adaptive beta for DPO, following beta-DPO (arXiv:2407.08639):
    the optimal beta tracks how informative the current batch's pairs are,
    which is measurable by the implicit reward margin. The controller input is
    the raw (beta-independent) margin that training already logs, so there is
    no feedback loop through beta itself.

    Per optimizer step: an EMA M0 of the margin is maintained, and
    beta = clamp(beta0 * (1 + alpha * tanh((M - M0) / s)), beta0/4, beta0*4)
    where s is the running standard deviation of the margin. The tanh bounds
    each step (beta-DPO's rule is linear and unbounded); the first
    warmup_steps updates return beta0 while M0 seeds.
    """

    def __init__(
            self,
            beta0: float,
            alpha: float = 0.5,
            ema_decay: float = 0.9,
            warmup_steps: int = 25,
    ):
        if beta0 <= 0:
            raise ValueError("beta0 must be positive")
        self.beta0 = beta0
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.min_beta = beta0 / 4.0
        self.max_beta = beta0 * 4.0

        self.margin_ema: float | None = None
        self._steps = 0
        self._running_mean = 0.0
        self._running_m2 = 0.0  # Welford accumulator for the tanh scale

    def update(self, margin: float) -> float:
        self._steps += 1

        delta = margin - self._running_mean
        self._running_mean += delta / self._steps
        self._running_m2 += delta * (margin - self._running_mean)

        if self.margin_ema is None:
            self.margin_ema = margin
        else:
            self.margin_ema = self.ema_decay * self.margin_ema + (1.0 - self.ema_decay) * margin

        if self._steps <= self.warmup_steps:
            return self.beta0

        std = math.sqrt(self._running_m2 / (self._steps - 1)) if self._steps > 1 else 0.0
        scale = std if std > 1e-12 else max(abs(self.margin_ema), 1e-12)
        beta = self.beta0 * (1.0 + self.alpha * math.tanh((margin - self.margin_ema) / scale))
        return min(self.max_beta, max(self.min_beta, beta))
