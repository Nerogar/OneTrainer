from collections.abc import Callable

from modules.module.quantized.LinearFp8 import LinearFp8
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.enum.DataType import DataType

import torch
from torch import Tensor, nn

try:
    from modules.module.quantized.LinearNf4 import LinearNf4

    import bitsandbytes as bnb
except ImportError:
    bnb = None
    LinearNf4 = None


def __create_nf4_linear_layer(module: nn.Linear, copy_parameters: bool) -> nn.Module:
    bias = module.bias is not None

    quant_linear = LinearNf4(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=bias,
    )

    if copy_parameters:
        quant_linear.weight.data = module.weight.data
        if bias:
            quant_linear.bias.data = module.bias.data

    return quant_linear


def __create_int8_linear_layer(module: nn.Linear, copy_parameters: bool) -> nn.Module:
    bias = module.bias is not None

    quant_linear = bnb.nn.Linear8bitLt(
        input_features=module.in_features,
        output_features=module.out_features,
        bias=bias,
        has_fp16_weights=False,
    )

    if copy_parameters:
        quant_linear.weight = type(quant_linear.weight)(module.weight)
        if bias:
            quant_linear.bias = type(quant_linear.bias)(module.bias)

    return quant_linear


def __create_fp8_linear_layer(module: nn.Linear, copy_parameters: bool) -> nn.Module:
    bias = module.bias is not None

    quant_linear = LinearFp8(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=bias,
    )

    if copy_parameters:
        quant_linear.weight = type(quant_linear.weight)(module.weight)
        if bias:
            quant_linear.bias = type(quant_linear.bias)(module.bias)

    return quant_linear


def __replace_linear_layers(
        parent_module: nn.Module,
        convert_fn: Callable[[nn.Linear, bool], nn.Module],
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
                quant_linear = convert_fn(module, copy_parameters)
                parent_module[i] = quant_linear
                del module
            elif id(module) not in visited_modules:
                __replace_linear_layers(
                    parent_module=module,
                    convert_fn=convert_fn,
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
                quant_linear = convert_fn(module, copy_parameters)
                setattr(parent_module, attr_name, quant_linear)
                del module
            elif isinstance(module, nn.Module) and id(module) not in visited_modules:
                __replace_linear_layers(
                    parent_module=module,
                    convert_fn=convert_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    copy_parameters=copy_parameters,
                    name_prefix=f"{name_prefix}.{attr_name}",
                    visited_modules=visited_modules,
                )

    for name, module in parent_module.named_modules():
        #ensure that all Linear layers were replaced
        #https://github.com/Nerogar/OneTrainer/issues/1050
        assert not isinstance(module, nn.Linear) or isinstance(module, QuantizedLinearMixin), f"Linear layer {name} was not found in model for quantization"

def replace_linear_with_nf4_layers(
        parent_module: nn.Module,
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
):
    __replace_linear_layers(
        parent_module=parent_module,
        convert_fn=__create_nf4_linear_layer,
        keep_in_fp32_modules=keep_in_fp32_modules,
        copy_parameters=copy_parameters,
    )


def replace_linear_with_int8_layers(
        parent_module: nn.Module,
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
):
    __replace_linear_layers(
        parent_module=parent_module,
        convert_fn=__create_int8_linear_layer,
        keep_in_fp32_modules=keep_in_fp32_modules,
        copy_parameters=copy_parameters,
    )


def replace_linear_with_fp8_layers(
        parent_module: nn.Module,
        keep_in_fp32_modules: list[str] | None = None,
        copy_parameters: bool = False,
):
    __replace_linear_layers(
        parent_module=parent_module,
        convert_fn=__create_fp8_linear_layer,
        keep_in_fp32_modules=keep_in_fp32_modules,
        copy_parameters=copy_parameters,
    )


def is_quantized_parameter(
        module: nn.Module,
        parameter_name: str,
) -> bool:
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

    if isinstance(module, LinearFp8):
        return parameter_name == "weight"

    return False


def quantize_layers(module: nn.Module, device: torch.device, train_dtype: DataType):
    if module is not None:
        for child_module in module.modules():
            if isinstance(child_module, QuantizedModuleMixin):
                child_module.compute_dtype = train_dtype.torch_dtype()
                child_module.quantize(device)


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
