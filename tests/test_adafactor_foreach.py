"""
Equivalence tests: OneTrainer's per-parameter Adafactor step vs the foreach step.

Run with:  python -m pytest tests/test_adafactor_foreach.py   (or plain: python tests/test_adafactor_foreach.py)
"""
import copy
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.util import bf16_stochastic_rounding
from modules.util.optimizer import adafactor_extensions
from modules.util.optimizer.adafactor_extensions import patch_adafactor, step_adafactor_foreach

import torch

from transformers import Adafactor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1D unfactored, 2D/3D/4D factored, plus LoRA-like pairs. Repeated shapes exercise the batched per-shape path.
SHAPES = [(96,), (48, 32), (6, 10, 12), (16, 8, 3, 3), (4, 320), (320, 4), (4, 320), (320, 4), (6, 10, 12), (96,)]

CONFIGS = {
    "fixed_lr": {"lr": 1e-3, "relative_step": False, "scale_parameter": False, "warmup_init": False, "beta1": None, "weight_decay": 0.0},
    "fixed_lr_beta1": {"lr": 1e-3, "relative_step": False, "scale_parameter": False, "warmup_init": False, "beta1": 0.9, "weight_decay": 0.0},
    "fixed_lr_wd": {"lr": 1e-3, "relative_step": False, "scale_parameter": False, "warmup_init": False, "beta1": 0.9, "weight_decay": 0.01},
    "scale_param": {"lr": 1e-3, "relative_step": False, "scale_parameter": True, "warmup_init": False, "beta1": None, "weight_decay": 0.01},
    "relative_step": {"lr": None, "relative_step": True, "scale_parameter": True, "warmup_init": False, "beta1": None, "weight_decay": 0.0},
    "relative_warmup": {"lr": None, "relative_step": True, "scale_parameter": True, "warmup_init": True, "beta1": 0.9, "weight_decay": 0.01},
}

DTYPES = [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True), (torch.float16, False)]

STATE_KEYS = ["step", "RMS", "exp_avg", "exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col"]

# Batched reductions and fused clip/LR scaling can differ slightly from the reference path. For 16 bit params this can
# occasionally move a value to a neighbouring representable value
TOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2, torch.float16: 1e-3}


def make_params(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g).mul_(0.1).to(device=DEVICE, dtype=dtype).requires_grad_(True) for s in SHAPES]


def make_optimizer(params, cfg, foreach, stochastic_rounding):
    opt = Adafactor(params, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, **cfg)
    patch_adafactor(opt, stochastic_rounding, foreach)
    return opt


def set_grads(params, step):
    g = torch.Generator().manual_seed(1000 + step)
    for i, p in enumerate(params):
        # skip one parameter on some steps, so step counts diverge between parameters
        if step % 3 == 1 and i == 2:
            p.grad = None
        else:
            p.grad = torch.randn(p.shape, generator=g).to(device=p.device, dtype=p.dtype)


def run_steps(params, opt, first, last):
    bf16_stochastic_rounding.set_seed(123 + first, DEVICE)
    for step in range(first, last):
        set_grads(params, step)
        opt.step()
    return params, opt


def run(dtype, cfg, foreach, stochastic_rounding, steps=6):
    params = make_params(dtype)
    return run_steps(params, make_optimizer(params, cfg, foreach, stochastic_rounding), 0, steps)


def normalized_max_abs_error(a, b):
    """max |a - b| over all elements, divided by max |a|"""
    if isinstance(a, torch.Tensor):
        a, b = a.float(), b.float()
        return ((a - b).abs().max() / a.abs().max().clamp(min=1e-30)).item()
    return abs(a - b) / max(abs(a), 1e-30)


def compare(ref, new):
    """checks matching state layout, returns the worst normalized max abs error over params and state tensors"""
    (p_ref, o_ref), (p_new, o_new) = ref, new
    worst = 0.0
    for a, b in zip(p_ref, p_new, strict=True):
        assert a.dtype == b.dtype
        worst = max(worst, normalized_max_abs_error(a, b))
        s_ref, s_new = o_ref.state[a], o_new.state[b]
        assert set(s_ref.keys()) == set(s_new.keys()), (s_ref.keys(), s_new.keys())
        assert s_ref["step"] == s_new["step"]
        for k in STATE_KEYS:
            if k in s_ref:
                assert type(s_ref[k]) is type(s_new[k]), k
                if isinstance(s_ref[k], torch.Tensor):
                    assert s_ref[k].shape == s_new[k].shape and s_ref[k].dtype == s_new[k].dtype, k
                worst = max(worst, normalized_max_abs_error(s_ref[k], s_new[k]))
    return worst


def bit_exact(ref, new):
    (p_ref, o_ref), (p_new, o_new) = ref, new
    for a, b in zip(p_ref, p_new, strict=True):
        if not torch.equal(a, b):
            return False
        for k in STATE_KEYS:
            v = o_ref.state[a].get(k)
            if isinstance(v, torch.Tensor) and not torch.equal(v, o_new.state[b][k]):
                return False
    return True


def check_case(cfg_name, dtype, sr):
    cfg = CONFIGS[cfg_name]
    ref, new = run(dtype, cfg, False, sr), run(dtype, cfg, True, sr)
    err = compare(ref, new)
    assert err <= TOL[dtype], f"{cfg_name} {dtype} sr={sr}: normalized max abs error {err}"
    return err, bit_exact(ref, new)


def test_equivalence():
    for cfg_name, (dtype, sr) in itertools.product(CONFIGS, DTYPES):
        check_case(cfg_name, dtype, sr)


def test_stochastic_rounding_stream():
    g = torch.Generator().manual_seed(7)
    sources = [torch.randn(s, generator=g).to(DEVICE) for s in SHAPES]
    flat = torch.cat([s.flatten() for s in sources])

    bf16_stochastic_rounding.set_seed(5, DEVICE)
    ref = [torch.empty(s, device=DEVICE, dtype=torch.bfloat16) for s in SHAPES]
    for t, s in zip(ref, sources, strict=True):
        bf16_stochastic_rounding.copy_stochastic_(t, s)
    ref_next = torch.randint(0, 1 << 30, (8,), device=DEVICE, generator=bf16_stochastic_rounding.generator)

    bf16_stochastic_rounding.set_seed(5, DEVICE)
    new = [torch.empty(s, device=DEVICE, dtype=torch.bfloat16) for s in SHAPES]
    bf16_stochastic_rounding.copy_stochastic_flat_(new, flat)
    new_next = torch.randint(0, 1 << 30, (8,), device=DEVICE, generator=bf16_stochastic_rounding.generator)

    assert all(torch.equal(a, b) for a, b in zip(ref, new, strict=True))
    assert torch.equal(ref_next, new_next), "generator state diverged"


def run_chunked(limit, dtype, sr):
    chunks = []
    orig_chunk, orig_limit = adafactor_extensions._step_adafactor_chunk, adafactor_extensions.FOREACH_CHUNK_NUMEL

    def recording_chunk(self, group, params):
        chunks.append([p.numel() for p in params])
        orig_chunk(self, group, params)

    adafactor_extensions._step_adafactor_chunk = recording_chunk
    adafactor_extensions.FOREACH_CHUNK_NUMEL = limit
    try:
        new = run(dtype, CONFIGS["fixed_lr_beta1"], True, sr)
    finally:
        adafactor_extensions._step_adafactor_chunk = orig_chunk
        adafactor_extensions.FOREACH_CHUNK_NUMEL = orig_limit
    return new, chunks


def test_chunking():
    # a chunk limit that splits between params (3000), and one smaller than all but one single param (100)
    worst = 0.0
    for limit, (dtype, sr) in itertools.product([3000, 100], [(torch.float32, False), (torch.bfloat16, True)]):
        new, chunks = run_chunked(limit, dtype, sr)
        assert all(sum(c) <= limit or len(c) == 1 for c in chunks), chunks
        if limit == 3000:
            assert any(len(c) > 1 for c in chunks), chunks
        else:
            assert any(c[0] > limit for c in chunks), chunks
        err = compare(run(dtype, CONFIGS["fixed_lr_beta1"], False, sr), new)
        assert err <= TOL[dtype], f"limit {limit} {dtype} sr={sr}: {err}"
        worst = max(worst, err)
    return worst


def test_checkpoint_resume():
    # a save by one implementation must resume in the other like a loop save + loop resume (load_state_dict casts the
    # state to the param dtype, so the reference has to go through it as well)
    cfg = CONFIGS["fixed_lr_wd"]

    def save_and_resume(dtype, sr, save_foreach, resume_foreach):
        params, opt = run(dtype, cfg, save_foreach, sr, steps=4)
        sd = copy.deepcopy(opt.state_dict())
        ps = [p.detach().clone().requires_grad_(True) for p in params]
        o = make_optimizer(ps, cfg, resume_foreach, sr)
        o.load_state_dict(sd)
        return run_steps(ps, o, 4, 8)

    worst = 0.0
    for (dtype, sr), save_foreach in itertools.product([(torch.float32, False), (torch.bfloat16, True)], [False, True]):
        ref = save_and_resume(dtype, sr, False, False)
        err = compare(ref, save_and_resume(dtype, sr, save_foreach, not save_foreach))
        assert err <= TOL[dtype], f"{dtype} sr={sr} saved by {'foreach' if save_foreach else 'loop'}: {err}"
        worst = max(worst, err)
    return worst


def test_mixed_devices():
    if DEVICE.type != "cuda":
        return None
    cfg = CONFIGS["fixed_lr_beta1"]

    def run_mixed(foreach):
        params = [p.detach().to("cpu" if i >= len(SHAPES) // 2 else DEVICE).requires_grad_(True)
                  for i, p in enumerate(make_params(torch.float32))]
        return run_steps(params, make_optimizer(params, cfg, foreach, False), 0, 6)

    err = compare(run_mixed(False), run_mixed(True))
    assert err <= TOL[torch.float32], f"mixed devices: {err}"
    return err


def test_fused_back_pass_rejects_foreach():
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.create import create_optimizer
    from modules.util.enum.Optimizer import Optimizer
    from modules.util.enum.TrainingMethod import TrainingMethod
    from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection

    def create(fused_back_pass, foreach):
        config = TrainConfig.default_values()
        config.training_method = TrainingMethod.LORA
        config.optimizer.optimizer = Optimizer.ADAFACTOR
        config.optimizer.relative_step = False
        config.optimizer.fused_back_pass = fused_back_pass
        config.optimizer.foreach = foreach
        groups = NamedParameterGroupCollection()
        groups.add_group(NamedParameterGroup("p", make_params(torch.float32), 1e-3))
        return create_optimizer(groups, None, config)

    try:
        create(fused_back_pass=True, foreach=True)
        raise AssertionError("fused_back_pass + foreach was accepted")
    except RuntimeError as e:
        assert "fused_back_pass" in str(e)
    assert create(fused_back_pass=False, foreach=True).step.__func__ is step_adafactor_foreach
    assert create(fused_back_pass=True, foreach=False).step.__func__ is not step_adafactor_foreach


if __name__ == "__main__":
    print(f"device: {DEVICE}, torch {torch.__version__}")
    print("error = max |loop - foreach| / max |loop|, worst over params and state tensors")
    for cfg_name, (dtype, sr) in itertools.product(CONFIGS, DTYPES):
        err, exact = check_case(cfg_name, dtype, sr)
        print(f"{cfg_name:16s} {str(dtype):15s} sr={sr!s:5s} error {err:.3e}  bit exact: {exact}")
    test_stochastic_rounding_stream()
    print("stochastic rounding: batched writeback bit identical to copy_stochastic_, same generator state")
    print(f"tiny chunks / oversized tensors  error {test_chunking():.3e}")
    print(f"checkpoint resume across impls   error {test_checkpoint_resume():.3e}")
    if (err := test_mixed_devices()) is not None:
        print(f"params split over cpu and cuda   error {err:.3e}")
    test_fused_back_pass_rejects_foreach()
    print("fused_back_pass + foreach rejected")
    print("OK")
