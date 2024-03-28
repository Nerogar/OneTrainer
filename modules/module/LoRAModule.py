import abc
import itertools
import math
import typing

import torch
from torch import nn, Tensor
from torch.nn import Dropout, Linear, Conv2d, Parameter

from modules.util.enum.LoraType import LoraType

ModuleDict = dict[str, nn.Module]
StateDict = dict[str, torch.Tensor]



class BaseLoRAModule(abc.ABC):
    type: typing.ClassVar[LoraType]
    orig_module: nn.Module
    is_applied: bool
    prefix: str
    
    def __init__(self, prefix: str, orig_module: nn.Module) -> None:
        super().__init__()
        self.prefix = prefix
        self.orig_module = orig_module
        self.orig_forward = orig_module.forward
        self.is_applied = False
    
    @abc.abstractmethod
    def grad_modules(self) -> ModuleDict:
        raise RuntimeError("Do not call `super().grad_modules`.")
    
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Do not call `super().forward`.")
    
    def requires_grad_(self, requires_grad: bool) -> "BaseLoRAModule":
        for module in self.grad_modules().values():
            module.requires_grad_(requires_grad)
        
    lora_down: nn.Module
    lora_up: nn.Module
    alpha: torch.Tensor
    dropout: Dropout

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super(LoRAModule, self).__init__()

        self.prefix = prefix.replace('.', '_')
        self.orig_module = orig_module
        self.rank = rank
        self.alpha = torch.tensor(alpha)
        self.dropout = Dropout(0)
        if orig_module is not None:
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

        self.is_applied = False
        self.orig_forward = self.orig_module.forward if self.orig_module is not None else None

    def forward(self, x, *args, **kwargs):
        if self.orig_module.training:
            ld = self.lora_up(self.dropout(self.lora_down(x)))
            return self.orig_forward(x) + ld * (self.alpha / self.rank)

        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

    def requires_grad_(self, requires_grad: bool):
        self.lora_down.requires_grad_(requires_grad)
        self.lora_up.requires_grad_(requires_grad)

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModule':
        self.lora_down.to(device, dtype)
        self.lora_up.to(device, dtype)
        self.alpha.to(device, dtype)
        return self
            
    def parameters(self) -> typing.Iterable[nn.Parameter]:
        return itertools.chain(*(module.parameters() for module in self.grad_modules().values()))
    
    @abc.abstractmethod
    def load_state_dict(self, state_dict: StateDict) -> typing.Self:
        for key, module in self.grad_modules().items():
            module_state_dict: StateDict = {}
            for name, _ in module.named_parameters():
                module_state_dict[name] = state_dict.pop(f"{self.prefix}.{key}.{name}")
            
            module.load_state_dict(module_state_dict)
        
        return self
                
    
    @abc.abstractmethod
    def state_dict(self) -> StateDict:
        params = {}
        
        for key, module in self.grad_modules().items():
            for param_key, param in module.state_dict().items():
                params[f"{self.prefix}.{key}.{param_key}"] = param
        
        return params
    
    def hook_to_module(self):
        if not self.is_applied:
            self.orig_module.forward = self.forward
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.is_applied = False
    
    def to(self, device: torch.device | None  = None, dtype: torch.dtype | None = None) -> typing.Self:
        for module in self.grad_modules().values():
            module.to(device=device, dtype=dtype)
        
        return self
            
    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass  
        

LORA_TYPE_MAP: dict[tuple[typing.Type[nn.Module], LoRAType], typing.Type[BaseLoRAModule]] = {}


class LoRAModule(BaseLoRAModule, abc.ABC):
    orig_module: nn.Module
    lora_down: nn.Module
    lora_up: nn.Module
    alpha: float

    def __init__(self, prefix: str, orig_module: nn.Module, rank: int, alpha: float):
        super(LoRAModule, self).__init__(prefix, orig_module)
        self.rank = rank
        self.alpha = alpha

    def forward(self, x, *args, **kwargs):
        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)
    
    def grad_modules(self) -> dict[str, nn.Module]:
        return {
            "lora_down": self.lora_down,
            "lora_up": self.lora_up,
        }

    def load_state_dict(self, state_dict: StateDict) -> typing.Self:
        super().load_state_dict(state_dict)
        self.alpha = state_dict.pop(f"{self.prefix}.alpha").item()
        return self

    def state_dict(self) -> StateDict:
        state_dict = super(LoRAModule, self).state_dict()
        state_dict[f"{self.prefix}.alpha"] = torch.tensor(self.alpha)
        return state_dict
        



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
            orig_module: nn.Module,
            rank: int,
            prefix: str,
            alpha: float = 1.0,
            module_filter: list[str] | None = None,
    ):
        super(LoRAModuleWrapper, self).__init__()
        self.orig_module = orig_module
        self.rank = rank
        self.prefix = prefix
        self.module_filter = module_filter

        self.modules = self.__create_modules(orig_module, alpha)

    def __create_modules(self, orig_module: nn.Module | None, alpha: float) -> dict[str, LoRAModule]:
        lora_modules = {}

        if orig_module is not None:
            for name, child_module in orig_module.named_modules():
                if self.module_filter is None or any([x in name for x in self.module_filter]):
                    if isinstance(child_module, Linear):
                        lora_modules[name] = LinearLoRAModule(self.prefix + "_" + name, child_module, self.rank, alpha)
                    elif isinstance(child_module, Conv2d):
                        lora_modules[name] = Conv2dLoRAModule(self.prefix + "_" + name, child_module, self.rank, alpha)

        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for module in self.modules.values():
            module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for module in self.modules.values():
            parameters += module.parameters()
        return parameters

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> 'LoRAModuleWrapper':
        for module in self.modules.values():
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

        for module in self.modules.values():
            state_dict |= module.state_dict()

        return state_dict

    def hook_to_module(self):
        """
        Hooks the LoRA into the module without changing its weights
        """
        for module in self.modules.values():
            module.hook_to_module()

    def remove_hook_from_module(self):
        """
        Removes the LoRA hook from the module without changing its weights
        """
        for module in self.modules.values():
            module.remove_hook_from_module()

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for module in self.modules.values():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for module in self.modules.values():
            module.extract_from_module(base_module)

    def prune(self):
        """
        Removes all dummy modules
        """
        self.modules = {k: v for (k, v) in self.modules.items() if not isinstance(v, DummyLoRAModule)}

    def set_dropout(self, dropout_probability: float):
        """
        Sets the dropout probability
        """
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError("Dropout probability must be in [0, 1]")
        for module in self.modules.values():
            module.dropout.p = dropout_probability
