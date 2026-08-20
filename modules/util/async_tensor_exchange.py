import torch


class AsyncTensorExchange:
    # Copies a device tensor into a pinned host buffer without synchronizing the calling
    # stream. exchange() returns the (value, key) passed to the *previous* call (or
    # (None, None) on the first call), by which point its async copy is guaranteed complete
    # - so the caller gets a value one call late, but reading it back costs a host-side wait
    # instead of a stream-draining sync. The key is opaque to this class; it's handed back
    # unchanged alongside the value so the caller doesn't have to track it separately.
    def __init__(self, device: torch.device, shape: tuple[int, ...] = (), dtype: torch.dtype = torch.float32):
        self.pin = torch.empty(shape, dtype=dtype, pin_memory=(device.type == "cuda"))
        self.event = torch.cuda.Event() if device.type == "cuda" else None
        self.pending_key = None

    def exchange(self, value: torch.Tensor, key):
        previous_value, previous_key = None, None
        if self.pending_key is not None:
            if self.event is not None:
                self.event.synchronize()
            # no event on non-CUDA devices, but there's also no async copy to wait for there:
            # non_blocking below is only True when self.event is set, so copy_() blocks until
            # done and self.pin is already fully written by the time we get here
            previous_value, previous_key = self.pin.clone(), self.pending_key

        self.pin.copy_(value, non_blocking=(self.event is not None))
        if self.event is not None:
            self.event.record()
        self.pending_key = key

        return previous_value, previous_key
