from abc import ABCMeta

import modules.util.compile_util as compile_util
import modules.util.nvcomp_util as nvcomp_util

import torch


class CompressedWeightMixin(metaclass=ABCMeta):
    def _init_compressed_state(self):
        self.compress = False
        self._compressed = False
        self._weight_shape = None
        self._uncompressed_bytes = 0
        self._compressed_dtype = None

    def _decompress(self, blob: torch.Tensor) -> torch.Tensor:
        # decoding only runs on the GPU. DoRA calls this during initialization, when the weight can
        # still be on the CPU: copy it to the GPU, decode there, and copy the result back.
        if blob.is_cuda:
            return nvcomp_util.decompress(blob, self._uncompressed_bytes, self._compressed_dtype, self._weight_shape)
        weight = nvcomp_util.decompress(blob.cuda(), self._uncompressed_bytes, self._compressed_dtype, self._weight_shape)
        return weight.to(device=blob.device)

    def uncompressed_bytes(self) -> int:
        # bytes the weight occupies decompressed; weight.nbytes is the stored size and drops to the
        # blob length once compressed
        if not self._compressed:
            return self.weight.nbytes
        return self._uncompressed_bytes

    @torch.no_grad()
    def _compress_weight(self, device: torch.device | None = None):
        if self._compressed:
            return

        weight = self.weight.detach()
        orig_device = weight.device
        gpu_weight = weight.to(device=device) if device is not None else weight
        if not gpu_weight.is_cuda:
            raise NotImplementedError("weight compression is CUDA-only")

        self._weight_shape = tuple(gpu_weight.shape)
        self._compressed_dtype = gpu_weight.dtype
        blob, self._uncompressed_bytes = nvcomp_util.compress(gpu_weight.contiguous())
        self._compressed = True

        if device is not None:
            blob = blob.to(device=orig_device)
        # the blob is non-float bytes and never a grad target; a grad-requiring Parameter
        # cannot hold a non-float .data, so drop grad before storing it
        self.weight.requires_grad_(False)
        self.weight.data = blob

        # compressed blobs vary in length per layer, so mark the length dynamic - otherwise
        # torch.compile compiles the block once per distinct compressed length. The flag re-applies
        # force_parameter_static_shapes in the reentrant-checkpoint backward worker threads
        # (pytorch#186537); the config set here only reaches this (main) thread. Both are required:
        # with force_parameter_static_shapes=True the mark is ignored outright for parameters.
        compile_util.needs_dynamic_parameter_shapes = True
        torch._dynamo.config.force_parameter_static_shapes = False
        torch._dynamo.decorators.maybe_mark_dynamic(self.weight, 0)
