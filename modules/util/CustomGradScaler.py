import torch
from torch.amp.grad_scaler import GradScaler, OptState


class DummyOptimizer:
    def __init__(self, parameter):
        self.param_groups = [
            {"params": [parameter]}
        ]


class CustomGradScaler(GradScaler):
    def __init__(self):
        super().__init__()

    def unscale_parameter_(self, parameter, optimizer):
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
        assert self._scale is not None
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            DummyOptimizer(parameter), inv_scale, found_inf, False
        )

    def maybe_opt_step_parameter(self, parameter, param_group, i, optimizer):
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            optimizer.step_parameter(parameter, param_group, i)

    def step_after_unscale_parameter_(self, optimizer):
        optimizer_state = self._per_optimizer_states[id(optimizer)]
        optimizer_state["stage"] = OptState.UNSCALED

        self._check_scale_growth_tracker("step")

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        assert (
                len(optimizer_state["found_inf_per_device"]) > 0
        ), "No inf checks were recorded for this optimizer."

        optimizer_state["stage"] = OptState.STEPPED
