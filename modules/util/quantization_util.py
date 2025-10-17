import os
from collections.abc import Callable
from functools import partial

from modules.module.quantized.LinearFp8 import LinearFp8
from modules.module.quantized.LinearSVD import BaseLinearSVD, make_svd_linear
from modules.module.quantized.LinearW8A8 import LinearW8A8
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType

import torch
from torch import Tensor, nn

from tqdm import tqdm

try:
    from modules.module.quantized.LinearNf4 import LinearNf4

    import bitsandbytes as bnb
except ImportError:
    bnb = None
    LinearNf4 = None

def __create_linear_layer(construct_fn, module: nn.Linear, copy_parameters: bool) -> nn.Module:
    bias = module.bias is not None

    quant_linear = construct_fn(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=bias,
    )

    if copy_parameters:
        quant_linear.weight = type(quant_linear.weight)(module.weight)
        if bias:
            quant_linear.bias = type(quant_linear.bias)(module.bias)

    return quant_linear


def __replace_linear_layers_recursive(
        parent_module: nn.Module,
        construct_fn,
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
        name_prefix: str = "",
        visited_modules: set[int] | None = None,
):
    if keep_in_fp32_modules is None:
        keep_in_fp32_modules = []

    # keeps track of all visited modules to prevent infinite recursion from cyclic graphs
    if visited_modules is None:
        visited_modules = set()

    visited_modules.add(id(parent_module))
    if isinstance(parent_module, (nn.ModuleList, nn.Sequential)):
        for i, module in enumerate(parent_module):
            if isinstance(module, nn.Linear):
                quant_linear = __create_linear_layer(construct_fn, module, copy_parameters)
                parent_module[i] = quant_linear
                del module
            elif id(module) not in visited_modules:
                __replace_linear_layers_recursive(
                    parent_module=module,
                    construct_fn=construct_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    copy_parameters=copy_parameters,
                    name_prefix=f"{name_prefix}[{i}]",
                    visited_modules=visited_modules,
                )
    else:
        for attr_name in list(dir(parent_module)):
            if attr_name in keep_in_fp32_modules:
                continue

            module = getattr(parent_module, attr_name)
            if isinstance(module, nn.Linear):
                quant_linear = __create_linear_layer(construct_fn, module, copy_parameters)
                setattr(parent_module, attr_name, quant_linear)
                del module
            elif isinstance(module, nn.Module) and id(module) not in visited_modules:
                __replace_linear_layers_recursive(
                    parent_module=module,
                    construct_fn=construct_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    copy_parameters=copy_parameters,
                    name_prefix=f"{name_prefix}.{attr_name}",
                    visited_modules=visited_modules,
                )

def __replace_linear_layers(
        parent_module: nn.Module,
        convert_fn: Callable[[nn.Linear, bool], nn.Module],
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
):
    __replace_linear_layers_recursive(parent_module, convert_fn, keep_in_fp32_modules, copy_parameters)

    #ensure that all Linear layers were replaced
    #https://github.com/Nerogar/OneTrainer/issues/1050
    for name, module in parent_module.named_modules():
        assert (not isinstance(module, nn.Linear)
                or isinstance(module, QuantizedLinearMixin)
                or any(s in name.split('.') for s in keep_in_fp32_modules)
               ), f"Linear layer {name} was not found in model for quantization"

def replace_linear_with_quantized_layers(
        parent_module: nn.Module,
        dtype: DataType,
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
):
    if dtype.quantize_nf4():
        construct_fn = make_svd_linear(LinearNf4) if dtype.quantize_svd() else LinearNf4
    elif dtype.quantize_int8():
        construct_fn = partial(make_svd_linear(bnb.nn.Linear8bitLt) if dtype.quantize_svd() else bnb.nn.Linear8bitLt, has_fp16_weights=False)
    elif dtype.quantize_fp8():
        construct_fn = make_svd_linear(LinearFp8) if dtype.quantize_svd() else LinearFp8
    elif dtype.quantize_intW8A8():
        construct_fn = partial(make_svd_linear(LinearW8A8) if dtype.quantize_svd() else LinearW8A8, dtype=torch.int8, compute_dtype=torch.bfloat16)
    elif dtype.quantize_fpW8A8():
        construct_fn = partial(make_svd_linear(LinearW8A8) if dtype.quantize_svd() else LinearW8A8, dtype=torch.float8_e4m3fn,  compute_dtype=torch.bfloat16)
    else:
        return

    __replace_linear_layers(
        parent_module=parent_module,
        construct_fn=construct_fn,
        keep_in_fp32_modules=keep_in_fp32_modules,
        copy_parameters=copy_parameters,
    )


def is_quantized_parameter(
        module: nn.Module,
        parameter_name: str,
) -> bool:
    if isinstance(module, BaseLinearSVD):
        if parameter_name in ["svd_up", "svd_down"]:
            return True
    if bnb is not None:
        if isinstance(module, LinearNf4):
            return parameter_name in [
                "weight",
                "absmax",
                "offset",
                "code",
                "nested_absmax",
                "nested_code",
            ]
        elif isinstance(module, bnb.nn.Linear8bitLt):
            return parameter_name == "weight"

    if isinstance(module, (LinearFp8, LinearW8A8)):
        return parameter_name == "weight"

    return False


def quantize_layers(module: nn.Module, device: torch.device, train_dtype: DataType, config: TrainConfig):
    if module is not None:
        cache_dir = config.cache_dir + "/quantization"
        os.makedirs(cache_dir, exist_ok=True)
        child_modules = list(module.modules())
        for child_module in tqdm(child_modules, desc="Quantizing model weights", total=len(child_modules), delay=5, smoothing=0.1):
            if isinstance(child_module, QuantizedModuleMixin):
                child_module.compute_dtype = train_dtype.torch_dtype()
                child_module.quantize(device=device, cache_dir=cache_dir, svd_dtype=config.svd_dtype.torch_dtype(), rank=config.svd_rank)


def get_unquantized_weight(module: nn.Module, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(module, QuantizedLinearMixin):
        return module.unquantized_weight(dtype, device)

    return module.weight.detach().to(dtype=dtype)


def get_weight_shape(module: nn.Module) -> torch.Size:
    param = module.weight

    if bnb is not None:
        if isinstance(module, LinearNf4):
            return module.shape

    return param.shape


def get_offload_tensors(module: nn.Module) -> list[torch.Tensor]:
    tensors = []

    if bnb is not None:
        if isinstance(module, LinearNf4):
            tensors += [module.quant_state.absmax]
    if isinstance(module, nn.Linear | nn.Conv2d):
        tensors += [module.weight]
    if isinstance(module, nn.Linear) and module.bias is not None:
        tensors += [module.bias]
    if isinstance(module, BaseLinearSVD):
        tensors += [module.svd_up]
        tensors += [module.svd_down]

    return tensors


def get_offload_tensor_bytes(module: nn.Module) -> int:
    tensors = get_offload_tensors(module)

    return sum(t.element_size() * t.numel() for t in tensors)


def offload_quantized(
        module: nn.Module,
        device: torch.device,
        non_blocking: bool = False,
        allocator: Callable[[torch.tensor], torch.tensor] | None = None,
):
    tensors = get_offload_tensors(module)

    if allocator is None:
        for tensor in tensors:
            tensor.data = tensor.data.to(device=device, non_blocking=non_blocking)
    else:
        for tensor in tensors:
            new_tensor = allocator(tensor)
            new_tensor.copy_(tensor.data, non_blocking=non_blocking)
            tensor.data = new_tensor
