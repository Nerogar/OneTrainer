import torch

from sympy import S


#code from https://github.com/pytorch/pytorch/blob/ed82d5fcfd80110565f69130f286c7bfec6db2dc/torch/utils/_sympy/functions.py#L481
#but accepts negative numbers, to avoid https://github.com/Nerogar/OneTrainer/issues/1126
#can be removed once https://github.com/pytorch/pytorch/pull/169726 is merged into a torch version we use
@classmethod
def Mod_patched_eval(cls, p, q):
    # This was adapted from: sympy/core/mod.py

    # Triggered by
    # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
    # assert p.is_integer, p
    # assert q.is_integer, q

    if q.is_zero:
        raise ZeroDivisionError("Modulo by zero")

    # Three cases:
    #   1. p == 0
    #   2. p is either q or -q
    #   3. p is integer and q == 1
    if p is S.Zero or p in (q, -q) or q == 1:
        return S.Zero

    # Evaluate if they are both literals.
    if q.is_Number and p.is_Number:
        if p < 0:
            #raise AssertionError(p)
            pass
        if q < 1:
            raise AssertionError(q)
        return p % q

    # If q == 2, it's a matter of whether p is odd or even.
    if q.is_Number and q == 2:
        if p.is_even:
            return S.Zero
        if p.is_odd:
            return S.One

    # If p is a multiple of q.
    r = p / q
    if r.is_integer:
        return S.Zero

    # If p < q and its ratio is positive, then:
    #   - floor(p / q) = 0
    #   - p % q = p - floor(p / q) * q = p
    less = p < q
    if less.is_Boolean and bool(less) and r.is_positive:
        return p

    return None


def init_compile():
    torch._dynamo.config.cache_size_limit = 8192
    torch.utils._sympy.functions.Mod.eval = Mod_patched_eval
