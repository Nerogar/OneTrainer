import math
from collections.abc import Callable

from modules.module.quantized.LinearFp8 import LinearFp8
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.enum.DataType import DataType

import torch
from torch import Tensor, nn

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def __create_nf4_linear_layer(module: nn.Linear) -> nn.Module:
    quant_linear = bnb.nn.LinearNF4(
        input_features=module.in_features,
        output_features=module.out_features,
        bias=module.bias is not None,
    )

    return quant_linear


def __create_int8_linear_layer(module: nn.Linear) -> nn.Module:
    quant_linear = bnb.nn.Linear8bitLt(
        input_features=module.in_features,
        output_features=module.out_features,
        bias=module.bias is not None,
        has_fp16_weights=False,
    )

    return quant_linear


def __create_fp8_linear_layer(module: nn.Linear) -> nn.Module:
    quant_linear = LinearFp8(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    )

    return quant_linear


def __replace_linear_layers(
        parent_module: nn.Module,
        convert_fn: Callable[[nn.Linear], nn.Module],
        keep_in_fp32_modules: list[str] | None = None,
        name_prefix: str = "",
        visited_modules: set[int] | None = None,
):
    if keep_in_fp32_modules is None:
        keep_in_fp32_modules = []

    # keeps track of all visited modules to prevent infinite recursion from cyclic graphs
    if visited_modules is None:
        visited_modules = set()

    visited_modules.add(id(parent_module))

    if isinstance(parent_module, nn.ModuleList):
        for i, module in enumerate(parent_module):
            if isinstance(module, nn.Linear):
                # print('replaced: ', f"{name_prefix}[{i}]")
                quant_linear = convert_fn(module)
                parent_module[i] = quant_linear
                del module
            elif id(module) not in visited_modules:
                __replace_linear_layers(
                    parent_module=module,
                    convert_fn=convert_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    name_prefix=f"{name_prefix}[{i}]",
                    visited_modules=visited_modules,
                )
    else:
        for attr_name in list(dir(parent_module)):
            if attr_name in keep_in_fp32_modules:
                continue

            module = getattr(parent_module, attr_name)
            if isinstance(module, nn.Linear):
                # print('replaced: ', f"{name_prefix}.{attr_name}")
                quant_linear = convert_fn(module)
                setattr(parent_module, attr_name, quant_linear)
                del module
            elif isinstance(module, nn.Module) and id(module) not in visited_modules:
                __replace_linear_layers(
                    parent_module=module,
                    convert_fn=convert_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    name_prefix=f"{name_prefix}.{attr_name}",
                    visited_modules=visited_modules,
                )


def replace_linear_with_nf4_layers(
        parent_module: nn.Linear,
        keep_in_fp32_modules: list[str] | None = None,
):
    __replace_linear_layers(
        parent_module=parent_module,
        convert_fn=__create_nf4_linear_layer,
        keep_in_fp32_modules=keep_in_fp32_modules,
    )


def replace_linear_with_int8_layers(
        parent_module: nn.Linear,
        keep_in_fp32_modules: list[str] | None = None,
):
    __replace_linear_layers(
        parent_module=parent_module,
        convert_fn=__create_int8_linear_layer,
        keep_in_fp32_modules=keep_in_fp32_modules,
    )


def replace_linear_with_fp8_layers(
        parent_module: nn.Linear,
        keep_in_fp32_modules: list[str] | None = None,
):
    __replace_linear_layers(
        parent_module=parent_module,
        convert_fn=__create_fp8_linear_layer,
        keep_in_fp32_modules=keep_in_fp32_modules,
    )


def is_quantized_parameter(
        module: nn.Module,
        parameter_name: str,
) -> bool:
    if bnb is not None:
        if isinstance(module, bnb.nn.LinearNF4 | bnb.nn.Linear8bitLt):
            return parameter_name == "weight"

    if isinstance(module, LinearFp8):
        return parameter_name == "weight"

    return False


def quantize_layers(module: nn.Module, device: torch.device | None = None):
    for child_module in module.modules():
        if isinstance(child_module, QuantizedModuleMixin):
            child_module.quantize(device)


def set_nf4_compute_type(module: nn.Module, dtype: DataType):
    for child_module in module.modules():
        if bnb is not None:
            if isinstance(child_module, bnb.nn.LinearNF4):
                child_module.compute_dtype = dtype.torch_dtype()
                child_module.compute_type_is_set = True
        if isinstance(child_module, LinearFp8):
            child_module.compute_dtype = dtype.torch_dtype()
            child_module.compute_type_is_set = True


def get_unquantized_weight(module: nn.Module, dtype: torch.dtype) -> Tensor:
    param = module.weight

    if bnb is not None:
        if isinstance(param, bnb.nn.Params4bit):
            if param.quant_state is not None:
                return bnb.functional.dequantize_4bit(
                    A=param.data,
                    quant_state=param.quant_state,
                    quant_type=param.quant_type,
                ).detach().to(dtype=dtype)
            else:
                return param.detach().to(dtype=dtype)
        if isinstance(param, bnb.nn.Int8Params):
            if param.dtype == torch.int8:  # already quantized
                if param.SCB is not None:
                    return (param.SCB.unsqueeze(1) * param.detach()) / 127
                else:  # SCB is saved in the module
                    return (module.state.SCB.unsqueeze(1) * param.detach()) / 127
            else:
                return param.detach().to(dtype=dtype)

    return param.detach().to(dtype=dtype)


def get_weight_shape(module: nn.Module) -> torch.Size:
    param = module.weight

    if bnb is not None:
        if isinstance(param, bnb.nn.Params4bit):
            if param.quant_state is not None:
                return param.quant_state.shape
            else:
                return param.shape

    return param.shape


def get_offload_tensors(module: nn.Module) -> list[torch.Tensor]:
    if isinstance(module, bnb.nn.Linear4bit | nn.Linear | nn.Conv2d):
        return [module.weight.data]
    return []


def get_offload_tensor_bytes(module: nn.Module) -> int:
    if isinstance(module, bnb.nn.Linear4bit):
        if module.weight.quant_state is None:
            return math.ceil(module.weight.numel() / 2)
        else:
            return module.weight.numel()
    if isinstance(module, nn.Linear | nn.Conv2d):
        return module.weight.element_size() * module.weight.numel()
    return 0


def offload_quantized(
        module: nn.Module,
        device: torch.device,
        non_blocking: bool = False,
        allocator: Callable[[torch.tensor], torch.tensor] | None = None,
):
    if allocator is None:
        if isinstance(module, bnb.nn.LinearNF4 | nn.Linear | nn.Conv2d):
            module.weight.data = module.weight.data.to(device=device, non_blocking=non_blocking)
    else:
        if isinstance(module, bnb.nn.LinearNF4 | nn.Linear | nn.Conv2d):
            tensor = allocator(module.weight)
            tensor.copy_(module.weight.data, non_blocking=non_blocking)
            module.weight.data = tensor
