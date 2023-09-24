import math
from abc import ABCMeta

import torch
from torch import nn, Tensor
from torch.nn import Linear, Conv2d, Parameter


class LoRAModule(metaclass=ABCMeta):
    prefix: str
    orig_module: nn.Module
    lora_down: nn.Module
    lora_up: nn.Module
    alpha: torch.Tensor

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super(LoRAModule, self).__init__()
        self.prefix = prefix.replace('.', '_')
        self.orig_module = orig_module
        self.rank = rank
        self.alpha = torch.tensor(alpha)
        if orig_module is not None:
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

        self.is_applied = False
        self.orig_forward = self.orig_module.forward if self.orig_module is not None else None

    def forward(self, x, *args, **kwargs):
        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

    def requires_grad_(self, requires_grad: bool):
        self.lora_down.requires_grad_(requires_grad)
        self.lora_up.requires_grad_(requires_grad)

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModule':
        self.lora_down.to(device, dtype)
        self.lora_up.to(device, dtype)
        self.alpha.to(device, dtype)
        return self

    def parameters(self) -> list[Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())

    def load_state_dict(self, state_dict: dict):
        down_state_dict = {
            "weight": state_dict.pop(self.prefix + ".lora_down.weight")
        }
        up_state_dict = {
            "weight": state_dict.pop(self.prefix + ".lora_up.weight")
        }
        self.alpha = state_dict.pop(self.prefix + ".alpha")

        self.lora_down.load_state_dict(down_state_dict)
        self.lora_up.load_state_dict(up_state_dict)

    def state_dict(self) -> dict:
        state_dict = {}
        state_dict[self.prefix + ".lora_down.weight"] = self.lora_down.weight.data
        state_dict[self.prefix + ".lora_up.weight"] = self.lora_up.weight.data
        state_dict[self.prefix + ".alpha"] = self.alpha
        return state_dict

    def hook_to_module(self):
        if not self.is_applied:
            self.orig_module.forward = self.forward
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.is_applied = False

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


class LinearLoRAModule(LoRAModule):
    def __init__(self, prefix: str, orig_module: Linear, rank: int, alpha: float):
        super(LinearLoRAModule, self).__init__(prefix, orig_module, rank, alpha)

        in_features = orig_module.in_features
        out_features = orig_module.out_features

        self.lora_down = Linear(in_features, rank, bias=False, device=orig_module.weight.device)
        self.lora_up = Linear(rank, out_features, bias=False, device=orig_module.weight.device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class Conv2dLoRAModule(LoRAModule):
    def __init__(self, prefix: str, orig_module: Conv2d, rank: int, alpha: float):
        super(Conv2dLoRAModule, self).__init__(prefix, orig_module, rank, alpha)
        in_channels = orig_module.in_channels
        out_channels = orig_module.out_channels

        self.lora_down = Conv2d(in_channels, rank, (1, 1), bias=False, device=orig_module.weight.device)
        self.lora_up = Conv2d(rank, out_channels, (1, 1), bias=False, device=orig_module.weight.device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class DummyLoRAModule(LoRAModule):
    def __init__(self, prefix: str):
        super(DummyLoRAModule, self).__init__(prefix, None, 1, 1)
        self.lora_down = None
        self.lora_up = None

        self.save_state_dict = {}

    def requires_grad_(self, requires_grad: bool):
        pass

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModule':
        pass

    def parameters(self) -> list[Parameter]:
        return []

    def load_state_dict(self, state_dict: dict):
        self.save_state_dict = {
            self.prefix + ".lora_down.weight": state_dict.pop(self.prefix + ".lora_down.weight"),
            self.prefix + ".lora_up.weight": state_dict.pop(self.prefix + ".lora_up.weight"),
            self.prefix + ".alpha": state_dict.pop(self.prefix + ".alpha"),
        }

    def state_dict(self) -> dict:
        return self.save_state_dict

    def hook_to_module(self):
        pass

    def remove_hook_from_module(self):
        pass

    def apply_to_module(self):
        pass

    def extract_from_module(self, base_module: nn.Module):
        pass


class LoRAModuleWrapper:
    orig_module: nn.Module
    rank: int

    modules: dict[str, LoRAModule]

    def __init__(
            self,
            orig_module: nn.Module | None,
            rank: int,
            prefix: str,
            alpha: float = 1.0,
            module_filter: list[str] = None,
    ):
        super(LoRAModuleWrapper, self).__init__()
        self.orig_module = orig_module
        self.rank = rank
        self.prefix = prefix
        self.module_filter = module_filter if module_filter is not None else []

        self.modules = self.__create_modules(orig_module, alpha)

    def __create_modules(self, orig_module: nn.Module | None, alpha: float) -> dict[str, LoRAModule]:
        lora_modules = {}

        if orig_module is not None:
            for name, child_module in orig_module.named_modules():
                if len(self.module_filter) == 0 or any([x in name for x in self.module_filter]):
                    if isinstance(child_module, Linear):
                        lora_modules[name] = LinearLoRAModule(self.prefix + "_" + name, child_module, self.rank, alpha)
                    elif isinstance(child_module, Conv2d):
                        lora_modules[name] = Conv2dLoRAModule(self.prefix + "_" + name, child_module, self.rank, alpha)

        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for name, module in self.modules.items():
            module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for name, module in self.modules.items():
            parameters += module.parameters()
        return parameters

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModuleWrapper':
        for name, module in self.modules.items():
            module.to(device, dtype)
        return self

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        """
        Loads the state dict

        Args:
            state_dict: the state dict
        """

        # create a copy, so the modules can pop states
        state_dict = {k: v for (k, v) in state_dict.items() if k.startswith(self.prefix)}

        for name, module in self.modules.items():
            module.load_state_dict(state_dict)

        # create dummy modules for the remaining keys
        remaining_names = list(state_dict.keys())
        for name in remaining_names:
            if name.endswith(".alpha"):
                prefix = name.removesuffix(".alpha")
                module = DummyLoRAModule(prefix)
                module.load_state_dict(state_dict)
                self.modules[prefix] = module

    def state_dict(self) -> dict:
        """
        Returns the state dict
        """
        state_dict = {}

        for name, module in self.modules.items():
            state_dict |= module.state_dict()

        return state_dict

    def hook_to_module(self):
        """
        Hooks the LoRA into the module without changing its weights
        """
        for name, module in self.modules.items():
            module.hook_to_module()

    def remove_hook_from_module(self):
        """
        Removes the LoRA hook from the module without changing its weights
        """
        for name, module in self.modules.items():
            module.remove_hook_from_module()

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for name, module in self.modules.items():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for name, module in self.modules.items():
            module.extract_from_module(base_module)

    def prune(self):
        """
        Removes all dummy modules
        """
        self.modules = {k: v for (k, v) in self.modules.items() if not isinstance(v, DummyLoRAModule)}
