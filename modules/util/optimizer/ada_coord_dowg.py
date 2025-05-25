# Based on the original version by the late community member SargeZT
import torch
import torch.optim


def upcast_stochastic_(t: torch.Tensor) -> torch.Tensor:
    rand = torch.randint_like(
        t,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    t_upcast = t.to(dtype=torch.float32, copy=True).view(dtype=torch.int32)
    t_upcast = t_upcast & 0xFFFF0000
    t_upcast = t_upcast | (rand & 0x0000FFFF)
    t_upcast = t_upcast.view(dtype=torch.float32)

    return t_upcast


class AdaCoordDoWG(torch.optim.Optimizer):
    def __init__(self, params, eps=1e-8, weight_decay=0.0, stochastic_rounding=False):
        defaults = dict(
            epsilon=eps,
            weight_decay=weight_decay,
            stochastic_rounding=stochastic_rounding,
        )
        super(AdaCoordDoWG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if "vt" not in state:
                    state["vt"] = torch.zeros_like(p.data, dtype=torch.bfloat16)

                vt = state["vt"]

                grad_bf16 = grad.to(dtype=torch.bfloat16)
                vt.addcmul_(grad_bf16, grad_bf16, value=epsilon)

                vt_sqrt = vt.sqrt().add_(epsilon)

                if group["stochastic_rounding"]:
                    denom = upcast_stochastic_(vt_sqrt)
                else:
                    denom = vt_sqrt.to(dtype=torch.float32)

                if group["stochastic_rounding"]:
                    grad_float = upcast_stochastic_(grad)
                else:
                    grad_float = grad.to(dtype=torch.float32)

                gt_hat = grad_float * epsilon

                p.data.addcdiv_(gt_hat, denom, value=-1.0)

                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"])

        return loss
