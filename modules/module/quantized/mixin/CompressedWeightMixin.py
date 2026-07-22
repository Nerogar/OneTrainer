from abc import ABCMeta

import modules.util.compile_util as compile_util
import modules.util.nvcomp_util as nvcomp_util

import torch


class CompressedWeightMixin(metaclass=ABCMeta):
    def _init_compressed_state(self):
        # compress (whether to compress at quantize time) is set by quantize_layers() from the
        # per-component config; default off so a layer built outside that path never compresses
        self.compress = False
        self._compressed = False
        self._weight_shape = None
        self._uncompressed_bytes = 0
        self._compressed_dtype = None

    def _decompress(self, blob: torch.Tensor) -> torch.Tensor:
        if not blob.is_cuda:
            raise NotImplementedError("compressed-weight decompression is CUDA-only")
        return nvcomp_util.decompress(blob, self._uncompressed_bytes, self._compressed_dtype, self._weight_shape)

    def compression_sizes(self) -> tuple[int, int] | None:
        # (uncompressed bytes, stored compressed bytes) once the weight is compressed, else None.
        # the compressed blob is the uint8 weight tensor, so its byte count is weight.numel().
        if not self._compressed:
            return None
        return self._uncompressed_bytes, self.weight.numel()

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
