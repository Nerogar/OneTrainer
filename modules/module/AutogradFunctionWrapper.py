import torch
from torch import Tensor


class AutogradFunctionWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, fwd, bwd, decode) -> Tensor:
        ctx.save_for_backward(weight)
        ctx.bwd = bwd
        ctx.decode = decode
        return fwd(x, decode(weight) if decode is not None else weight)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("the wrapped weight cannot be trained")
        (weight,) = ctx.saved_tensors
        if ctx.decode is not None:
            weight = ctx.decode(weight)
        return ctx.bwd(grad_output, weight), None, None, None, None
