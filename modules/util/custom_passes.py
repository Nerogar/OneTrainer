import torch
import torch.nn as nn
import torch.nn.functional as F


class LohaWeight(torch.autograd.Function):
    """Forward and backward passes for LoHa.

    LoHa has four matrices: two downs and two ups. The corresponding downs and
    ups are multiplied by normal matrix multiplication; the result is then
    multipled via the Hadamard product.

    This class contains the forward and backward passes for this specific
    operation (NOT forward/backward for Hadamard product in general).
    """

    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale=torch.tensor(1)) -> torch.Tensor:
        ctx.save_for_backward(w1d, w1u, w2d, w2u, scale)
        diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1d, w1u, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2u @ w2d)
        grad_w1u = temp @ w1d.T
        grad_w1d = w1u.T @ temp

        temp = grad_out * (w1u @ w1d)
        grad_w2u = temp @ w2d.T
        grad_w2d = w2u.T @ temp

        del temp
        return grad_w1d, grad_w1u, grad_w2d, grad_w2u, None
