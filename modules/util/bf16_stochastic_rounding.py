import torch
from torch import Tensor

generator = None

def set_seed(seed: int, device: torch.device):
    global generator
    if generator is None or generator.device != device:
        generator = torch.Generator(device=device)
    generator.manual_seed(seed)

def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """

    global generator

    # create a random 16 bit integer
    result = torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=generator,
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    adds other to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)

    result.add_(input, alpha=alpha)
    copy_stochastic_(input, result)


def addcdiv_stochastic_(input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0):
    """
    adds (tensor1 / tensor2 * value) to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        tensor1: the numerator tensor
        tensor2: the denominator tensor
        value: a multiplier for tensor1/tensor2
    """
    result = input.clone() if input.dtype == torch.float32 else input.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(input, result)
