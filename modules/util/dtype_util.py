from contextlib import nullcontext

from modules.util.enum.DataType import DataType

import torch
from torch.nn import Parameter


def enable_grad_scaling(train_dtype: DataType, parameters: list[Parameter]):
    trainable_parameter_dtype = list({parameter.dtype for parameter in parameters})
    return train_dtype == DataType.FLOAT_16 and all(dtype == torch.float32 for dtype in trainable_parameter_dtype)


def create_grad_scaler():
    from modules.util.CustomGradScaler import CustomGradScaler
    return CustomGradScaler()


def create_autocast_context(
        device: torch.device,
        train_dtype: DataType | None,
        enable_autocast_cache: bool,
) -> tuple[torch.autocast | nullcontext, DataType]:
    torch_train_dtype = train_dtype.torch_dtype()

    if torch_train_dtype in (torch.float16, torch.bfloat16):
        # fp16/bf16 autocast is supported on every backend. autocast casts the operands
        # of matmul/conv-type ops to train_dtype (precision-sensitive ops like norms stay
        # in fp32), so a weight stored at a different dtype is cast on the fly rather than
        # mismatching in the matmul.
        if device.type != "cuda":
            # CUDA (incl. ROCm) is the tested backend. fp16/bf16 autocast works on
            # other backends too (mps, xpu, cpu, ...) but is untested here; bf16 on
            # MPS additionally needs macOS >= 14.
            print(f"Warning: Mixed precision training is untested on device type '{device.type}'.")
        return torch.autocast(device_type=device.type, dtype=torch_train_dtype,
                              cache_enabled=enable_autocast_cache), train_dtype
    elif device.type == "cuda":
        # float32/tfloat32 on CUDA (and ROCm, which also reports device type "cuda"):
        # CUDA accepts float32 as an autocast dtype and upcasts lower-precision weights
        # on the fly (this is undocumented but works).
        return torch.autocast(device_type=device.type, dtype=torch_train_dtype,
                              cache_enabled=enable_autocast_cache), train_dtype
    else:
        # float32/tfloat32 on a non-CUDA backend (cpu, mps, xpu, ...): those backends
        # reject fp32 autocast, so disable autocast and let the model run at its weight
        # dtype. Disable explicitly (not nullcontext) so any enclosing autocast is
        # suppressed too.
        print("Warning: float32 training does not upcast lower-precision weights on this device "
              "(only CUDA can autocast to float32); the model runs at its weight dtype. "
              "Set the weight data types to float32 for full precision.")
        return torch.autocast(device_type=device.type, enabled=False), train_dtype


def disable_fp16_autocast_context(
        device: torch.device,
        train_dtype: DataType | None,
        fallback_train_dtype: DataType | None,
        enable_autocast_cache: bool,
) -> tuple[torch.autocast | nullcontext, DataType]:
    if train_dtype != DataType.FLOAT_16:
        # the main autocast context isn't fp16 -> nothing to override, defer to it
        return nullcontext(), train_dtype

    # fp16 training but this component is unstable in fp16 -> override the outer fp16
    # autocast and run it at the fallback precision. A bf16 fallback works on every
    # backend; a float32 fallback can only be applied via autocast on CUDA, so on
    # other devices disable autocast and let the component run at its weight dtype.
    fallback_torch_dtype = fallback_train_dtype.torch_dtype()
    if fallback_torch_dtype in (torch.float16, torch.bfloat16) or device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=fallback_torch_dtype,
                              cache_enabled=enable_autocast_cache), fallback_train_dtype
    else:
        print("Warning: the float32 fallback for fp16-unstable layers is not applied on device type "
              f"'{device.type}' (only CUDA can autocast to float32); these layers run at their weight dtype.")
        return torch.autocast(device_type=device.type, enabled=False), fallback_train_dtype


def disable_bf16_on_fp16_autocast_context(
        device: torch.device,
        train_dtype: DataType | None,
        weight_dtypes: list[DataType | None],
        enable_autocast_cache: bool,
) -> tuple[torch.autocast | nullcontext, DataType]:
    # Only used for the Wuerstchen / Stable Cascade effnet encoder. The rationale for
    # this special case is unknown, so its original behavior is deliberately kept
    # unchanged rather than migrated to the create_autocast_context approach above.
    weight_dtypes = list(filter(lambda dtype: dtype != DataType.NONE and dtype is not None, weight_dtypes))
    weight_dtypes = list(set(weight_dtypes))

    if all(weight_dtype != DataType.FLOAT_16 for weight_dtype in weight_dtypes):
        # weights are not in fp16 -> nothing to disable
        return nullcontext(), train_dtype

    if train_dtype != DataType.BFLOAT_16:
        # train dtype is not bf16 -> nothing to disable
        return nullcontext(), train_dtype

    if len(weight_dtypes) == 1:
        # all weights use the same dtype -> disable autocast
        return torch.autocast(device_type=device.type, enabled=False), weight_dtypes[0]

    return torch.autocast(device_type=device.type, dtype=weight_dtypes[0],
                          cache_enabled=enable_autocast_cache), weight_dtypes[0]
