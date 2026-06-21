from collections.abc import Callable

from modules.util.torch_util import torch_gc

from torch import nn


# A persistent delegating proxy for a module loaded on demand and discarded
# (weights freed) after use. Always truthy, so presence gates and captured
# references (e.g. MGDS Encode*Text nodes) stay valid while no weights are loaded.
# On-demand modules are frozen and inference-only by definition, so the proxy
# enforces that invariant: eval/requires_grad_(False) are accepted, while
# train(True)/requires_grad_(True) raise. Other attribute access delegates to the
# inner. The loader loads to cpu; the caller (model) is responsible for any
# quantization and for moving the materialized module to the accelerator.
#
# Intentionally a plain object, not an nn.Module, to avoid submodule/parameter
# registration fighting the materialize/discard swap.
class OnDemandModule:
    def __init__(self, loader: Callable[[], nn.Module]):
        self._inner: nn.Module | None = None
        self._loader = loader

    def __bool__(self):
        return True

    @property
    def inner(self) -> nn.Module | None:
        return self._inner

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        if self._inner is None:
            raise AttributeError(
                f"OnDemandModule has no materialized inner module; cannot access '{name}'"
            )
        return getattr(self._inner, name)

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("OnDemandModule is inference-only; train mode is not allowed")
        return self

    def requires_grad_(self, requires_grad: bool = True):
        if requires_grad:
            raise RuntimeError("OnDemandModule is frozen; requires_grad_(True) is not allowed")
        return self

    # No-op: the materialized module is placed on the accelerator by the caller and
    # freed by discard(). A blanket model.to(temp_device) must never move or reload it.
    def to(self, *args, **kwargs):
        return self

    def materialize(self):
        if self._inner is None:
            self._inner = self._loader()
            self._inner.eval()
            self._inner.requires_grad_(False)

    def discard(self):
        self._inner = None
        torch_gc()
