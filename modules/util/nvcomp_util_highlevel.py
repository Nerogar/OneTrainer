import torch

from nvidia import nvcomp  # raises ImportError if not installed, so nvcomp_util falls through

ALGORITHM = "ANS"

_codecs = {}       # stream pointer -> Codec (bound to that stream)
_decomp_configs = {}  # (id(codec), uncompressed_bytes) -> DecompressConfig


def _codec_for_stream(stream: torch.cuda.Stream):
    key = stream.cuda_stream
    codec = _codecs.get(key)
    if codec is None:
        codec = nvcomp.Codec(algorithm=ALGORITHM, cuda_stream=key)
        _codecs[key] = codec
    return codec


def _decomp_config_for(codec, uncompressed_bytes: int):
    key = (id(codec), uncompressed_bytes)
    config = _decomp_configs.get(key)
    if config is None:
        comp_config = codec.compression_config(uncompressed_bytes)  # size-only, sync-free
        config = codec.decompression_config(comp_config)            # derived, sync-free
        _decomp_configs[key] = config
    return config


@torch.no_grad()
def compress(weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    assert weight.is_cuda and weight.is_contiguous()
    stream = torch.cuda.current_stream(weight.device)
    codec = _codec_for_stream(stream)

    flat = weight.view(torch.uint8).reshape(-1)
    uncompressed_bytes = flat.numel()
    comp_config = codec.compression_config(uncompressed_bytes)
    src = nvcomp.as_array(flat, cuda_stream=stream.cuda_stream)
    encoded = codec.encode(src, compression_config=comp_config)
    # encode's output is backed by a max-size buffer and to_dlpack exports that whole capacity. Keep
    # only the actual compressed prefix (encoded.buffer_size) -- otherwise the stored weight is
    # LARGER than the uncompressed int8 weight and compression costs VRAM instead of saving it.
    n_comp = encoded.buffer_size
    compressed = torch.from_dlpack(encoded.to_dlpack(cuda_stream=stream.cuda_stream))[:n_comp].clone()
    return compressed, uncompressed_bytes


def decompress_into(compressed: torch.Tensor, out: torch.Tensor):
    # Sync-free decode on the current compute stream, writing into the caller's `out` buffer (its
    # size is the uncompressed length).
    stream = torch.cuda.current_stream(compressed.device)
    codec = _codec_for_stream(stream)
    config = _decomp_config_for(codec, out.numel())
    src = nvcomp.as_array(compressed, cuda_stream=stream.cuda_stream)
    codec.decode(src, decompression_config=config, out=nvcomp.as_array(out, cuda_stream=stream.cuda_stream))
