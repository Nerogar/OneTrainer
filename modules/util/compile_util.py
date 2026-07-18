import torch
import torch._dynamo.callback
import torch.utils._sympy.functions

from sympy import S
from tqdm import tqdm


#code from https://github.com/pytorch/pytorch/blob/ed82d5fcfd80110565f69130f286c7bfec6db2dc/torch/utils/_sympy/functions.py#L481
#but accepts negative numbers, to avoid https://github.com/Nerogar/OneTrainer/issues/1126
#torch 2.12 merged the fix PR https://github.com/pytorch/pytorch/pull/169726, but according to Claude's analysis this bug
#still exists. TODO confirm manually by testing https://github.com/Nerogar/OneTrainer/issues/1126:
#Claude: "Mod.eval itself still raises on negative numbers, and inductor's tiling_utils still calls sympy is_constant() (which
#substitutes negative test values) on expressions that can contain Mod. can be removed once Mod.eval itself
#accepts negative numbers in a torch version we use"
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
    # cache_size_limit and recompile_limit are aliases for the same dynamo config value.
    # since torch 2.12, dynamo config overrides are stored in ContextVars (pytorch/pytorch#173568),
    # which threads don't inherit. init_compile() must therefore be called in every thread that can
    # trigger a compilation - including the autograd worker threads, which recompute checkpointed
    # modules during the backward pass.
    torch._dynamo.config.cache_size_limit = 8192


def _on_compile_start(args: "torch._dynamo.callback.CallbackArgs") -> None:
    frame_id, _, frame_compile_id = args.compile_id.partition("/")
    direction = "backward" if args.callback_trigger == torch._dynamo.callback.CallbackTrigger.LAZY_BACKWARD else "forward"
    tqdm.write(f"[torch.compile] compiling kernel {frame_id} {direction} (variant #{frame_compile_id or 0})...")


torch._dynamo.callback.on_compile_start(_on_compile_start)

torch.utils._sympy.functions.Mod.eval = Mod_patched_eval
