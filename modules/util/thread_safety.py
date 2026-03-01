import functools
import threading

import torch

_THREAD_SAFE_FORWARD_ATTR = "_thread_safe_forward_lock"


def apply_thread_safe_forward(model: torch.nn.Module) -> None:
    """
    Wrap ``model.forward()`` with a per-instance ``threading.Lock`` to
    serialize concurrent calls.

    This is a workaround for a thread-safety bug in the transformers library's
    ``check_model_inputs`` decorator, which monkey-patches child module
    ``.forward()`` methods during execution and is not safe for concurrent use
    from multiple dataloader threads.

    See: https://github.com/huggingface/transformers/issues/42673
    Fix: https://github.com/huggingface/transformers/pull/43765 (v5 only)

    This patch can be removed when upgrading to transformers v5+.

    The lock is per-model-instance so different model instances do not block
    each other. The function is idempotent: calling it twice on the same model
    is a no-op.

    Args:
        model: The ``nn.Module`` whose ``forward()`` should be made thread-safe.
    """
    if hasattr(model, _THREAD_SAFE_FORWARD_ATTR):
        return

    lock = threading.Lock()
    original_forward = model.forward

    @functools.wraps(original_forward)
    def locked_forward(*args, **kwargs):
        with lock:
            return original_forward(*args, **kwargs)

    model.forward = locked_forward
    setattr(model, _THREAD_SAFE_FORWARD_ATTR, lock)
