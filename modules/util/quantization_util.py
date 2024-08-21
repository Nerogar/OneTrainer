from modules.util.enum.DataType import DataType

import torch
from torch import nn

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit


class LinearCompatibleNF4(Linear4bit):
    """
    Overwritten Linear4bit with torch.nn.Linear compatible parameters
    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            None,
            True,
            "nf4",
            torch.uint8,
            device,
        )


def __create_linear_with_nf4_layers(module: nn.Module):
    quant_linear = bnb.nn.LinearNF4(
        input_features=module.in_features,
        output_features=module.out_features,
        bias=module.bias is not None,
    )

    return quant_linear


def replace_linear_with_nf4_layers(
        parent_module: nn.Module,
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
                quant_linear = __create_linear_with_nf4_layers(module)
                parent_module[i] = quant_linear
                del module
            elif id(module) not in visited_modules:
                replace_linear_with_nf4_layers(
                    parent_module=module,
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
                quant_linear = __create_linear_with_nf4_layers(module)
                setattr(parent_module, attr_name, quant_linear)
                del module
            elif isinstance(module, nn.Module) and id(module) not in visited_modules:
                replace_linear_with_nf4_layers(
                    parent_module=module,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    name_prefix=f"{name_prefix}.{attr_name}",
                    visited_modules=visited_modules,
                )


def set_nf4_compute_type(module: nn.Module, dtype: DataType):
    for child_module in module.modules():
        if isinstance(child_module, bnb.nn.LinearNF4):
            child_module.compute_dtype = dtype.torch_dtype()
            child_module.compute_type_is_set = True
