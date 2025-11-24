from collections.abc import Callable
from functools import partial

from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.ModuleFilter import ModuleFilter

import torch
from torch import Tensor, nn

from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor

from tqdm import tqdm

try:
    from modules.module.quantized.LinearNf4 import LinearNf4

    import bitsandbytes as bnb
except ImportError:
    bnb = None
    LinearNf4 = None

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    q = x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q

def quantize_int8_tensorwise_get_scale(x: Tensor) -> float:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return scale

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, float]:
    scale = quantize_int8_tensorwise_get_scale(x)
    q = quantize_int8(x, scale)
    return q, scale

def quantize_int8_axiswise_get_scale(x: Tensor, dim: int) -> Tensor:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    scale = quantize_int8_axiswise_get_scale(x, dim)
    q = quantize_int8(x, scale)
    return q, scale

def quantize_fp8(x: Tensor, scale: float | Tensor) -> Tensor:
    q = x.float().mul(1.0 / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q

def quantize_fp8_tensorwise_get_scale(x: Tensor) -> float:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 448.0).clamp(min=1e-30)
    return scale

def quantize_fp8_axiswise_get_scale(x: Tensor, dim: int) -> Tensor:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 448.0).clamp(min=1e-30)
    return scale

def quantize_fp8_tensorwise(x: Tensor) -> tuple[Tensor, float]:
    scale = quantize_fp8_tensorwise_get_scale(x)
    q = quantize_fp8(x, scale)
    return q, scale

def quantize_fp8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    scale = quantize_fp8_axiswise_get_scale(x, dim)
    q = quantize_fp8(x, scale)
    return q, scale

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    return q.float() * scale


from modules.module.quantized.LinearFp8 import LinearFp8
from modules.module.quantized.LinearGGUFA8 import LinearGGUFA8
from modules.module.quantized.LinearSVD import BaseLinearSVD, make_svd_linear
from modules.module.quantized.LinearW8A8 import LinearW8A8


def __create_linear_layer(construct_fn, module: nn.Linear, copy_parameters: bool) -> nn.Module:
    bias = module.bias is not None
    quant_linear = construct_fn(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=bias,
    )

    if copy_parameters:
        quant_linear.weight = type(quant_linear.weight)(module.weight, requires_grad=False)
        if bias:
            quant_linear.bias = type(quant_linear.bias)(module.bias, requires_grad=False)

    return quant_linear



def __replace_linear_layers(
        parent_module: nn.Module,
        construct_fn,
        keep_in_fp32_modules: list[str] | None = None,
        filters: list[ModuleFilter] | None = None,
        copy_parameters: bool = False,
        name_prefix: str = "",
        visited_modules: set[int] | None = None,
        convert_type = nn.Linear,
):
    #both 'keep_in_fp32_modules' and 'filters' are layer filters: keep_in_fp32_modules is set by diffusers, 'filters' is set by the user.
    #Apply both. 'keep_in_fp32_modules' only looks at attr_name, 'filters' looks at the entire key at the leafs:
    if keep_in_fp32_modules is None:
        keep_in_fp32_modules = []

    # keeps track of all visited modules to prevent infinite recursion from cyclic graphs
    if visited_modules is None:
        visited_modules = set()

    visited_modules.add(id(parent_module))

    if isinstance(parent_module, (nn.ModuleList, nn.Sequential)):
        for i, module in enumerate(parent_module):
            if isinstance(module, convert_type):
                if filters is not None and len(filters) > 0 and not any(f.matches(name_prefix) for f in filters):
                    continue

                quant_linear = __create_linear_layer(construct_fn, module, copy_parameters)
                parent_module[i] = quant_linear
                del module
            elif id(module) not in visited_modules:
                __replace_linear_layers(
                    parent_module=module,
                    construct_fn=construct_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    filters=filters,
                    copy_parameters=copy_parameters,
                    name_prefix=f"{name_prefix}[{i}]",
                    visited_modules=visited_modules,
                )
    else:
        for attr_name in list(dir(parent_module)):
            if attr_name in keep_in_fp32_modules:
                continue

            module = getattr(parent_module, attr_name)
            if isinstance(module, convert_type):
                key_name = attr_name if name_prefix == "" else f"{name_prefix}.{attr_name}"
                if filters is not None and len(filters) > 0 and not any(f.matches(key_name) for f in filters):
                    continue

                quant_linear = __create_linear_layer(construct_fn, module, copy_parameters)
                setattr(parent_module, attr_name, quant_linear)
                del module
            elif isinstance(module, nn.Module) and id(module) not in visited_modules:
                __replace_linear_layers(
                    parent_module=module,
                    construct_fn=construct_fn,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    filters=filters,
                    copy_parameters=copy_parameters,
                    name_prefix=attr_name if name_prefix == "" else f"{name_prefix}.{attr_name}",
                    visited_modules=visited_modules,
                )

def replace_linear_with_quantized_layers(
        parent_module: nn.Module,
        dtype: DataType,
        keep_in_fp32_modules: list[str] | None = None,
        quantization: QuantizationConfig | None = None,
        copy_parameters: bool = False,
):
    kwargs = {}
    if dtype.quantize_nf4():
        linear_class = LinearNf4
    elif dtype.quantize_int8():
        linear_class = bnb.nn.Linear8bitLt
        kwargs = {'has_fp16_weights': False}
    elif dtype.quantize_fp8():
        linear_class = LinearFp8
    elif dtype.quantize_intW8A8():
        linear_class = LinearW8A8
        kwargs = {'dtype': torch.int8}
    elif dtype.quantize_fpW8A8():
        linear_class=LinearW8A8
        kwargs = {'dtype': torch.float8_e4m3fn}
    elif dtype == DataType.GGUF_A8_INT:
        linear_class=LinearGGUFA8
        kwargs = {'dtype': torch.int8}
    elif dtype == DataType.GGUF_A8_FLOAT:
        linear_class=LinearGGUFA8
        kwargs = {'dtype': torch.float8_e4m3fn}
    else:
        return

    if quantization is not None:
        if quantization.svd_dtype != DataType.NONE:
            if dtype.is_gguf():
                raise ValueError("SVDQuant cannot be used with GGUF. GGUF is loaded pre-quantized from a file. SVDQuant requires the unquantized weights to be available.")
            #construct_fn = partial(make_svd_linear(construct_fn), rank=quantization.svd_rank, svd_dtype=quantization.svd_dtype.torch_dtype(), cache_dir=quantization.cache_dir)
            linear_class = make_svd_linear(linear_class)
            kwargs.update({'rank': quantization.svd_rank,
                           'svd_dtype': quantization.svd_dtype.torch_dtype(),
                           'cache_dir': quantization.cache_dir})
        quant_filters = [
            ModuleFilter(pattern, use_regex=quantization.layer_filter_regex)
            for pattern in quantization.layer_filter.split(",")
        ]
    else:
        quant_filters = None

    convert_type = GGUFLinear if dtype.is_gguf() else nn.Linear
    __replace_linear_layers(
        parent_module=parent_module,
        construct_fn=partial(linear_class, **kwargs),
        keep_in_fp32_modules=keep_in_fp32_modules,
        filters=quant_filters,
        copy_parameters=copy_parameters,
        convert_type=convert_type,
    )

    #ensure that all Linear layers were replaced
    #https://github.com/Nerogar/OneTrainer/issues/1050
    for name, module in parent_module.named_modules():
        assert (not isinstance(module, convert_type)
                or isinstance(module, (QuantizedLinearMixin, LinearGGUFA8))
                or any(s in name.split('.') for s in keep_in_fp32_modules)
                or (quant_filters is not None and len(quant_filters) > 0 and not any(f.matches(name) for f in quant_filters))
               ), f"Linear layer {name} was not found in model for quantization"

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
        child_modules = list(module.modules())
        for child_module in tqdm(child_modules, desc="Quantizing model weights", total=len(child_modules), delay=5, smoothing=0.1):
            if isinstance(child_module, (QuantizedModuleMixin, GGUFLinear)):
                child_module.compute_dtype = train_dtype.torch_dtype()
            if isinstance(child_module, QuantizedModuleMixin):
                child_module.quantize(device=device)

def get_unquantized_weight(module: nn.Linear, dtype: torch.dtype, device: torch.device) -> Tensor:
    assert isinstance(module, nn.Linear)
    if isinstance(module, QuantizedLinearMixin):
        return module.unquantized_weight(dtype, device)
    elif isinstance(module, GGUFLinear):
        return dequantize_gguf_tensor(module.weight).to(dtype=dtype)
    else:
        return module.weight.detach().to(dtype=dtype)


def get_weight_shape(module: nn.Linear) -> torch.Size:
    assert isinstance(module, nn.Linear)
    return torch.Size((module.out_features, module.in_features))

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
