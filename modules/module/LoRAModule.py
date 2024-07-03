import math
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Dropout, Linear, Conv2d, Parameter

from modules.util import custom_passes


class PeftBase(metaclass=ABCMeta):
    is_applied: bool
    orig_forward: Any | None
    orig_module: nn.Module | None
    prefix: str
    layer_kwargs: dict
    op: Callable[[Tensor, Tensor], Tensor] | None

    def __init__(self, prefix: str, orig_module: nn.Module | None, layer_kwargs: dict = {}):
        self.prefix = prefix
        self.orig_module = orig_module
        self.is_applied = False
        self.layer_kwargs = layer_kwargs

        # For modules that have a custom backward pass and use functional ops.
        if orig_module is not None:
            match orig_module:
                case nn.Linear():
                    self.op = F.linear
                case nn.Conv2d():
                    self.op = F.conv2d
                case _:
                    raise NotImplementedError("Only Linear and Conv2d are supported layers.")

    def hook_to_module(self):
        assert self.orig_module

        if not self.is_applied:
            self.orig_module.forward = self.forward
            self.is_applied = True

    def remove_hook_from_module(self):
        assert self.orig_forward
        assert self.orig_module

        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.is_applied = False

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def forward(self, x, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def requires_grad_(self, requires_grad: bool):
        pass

    @abstractmethod
    def to(self, device: torch.device|None = None, dtype: torch.dtype|None = None) -> Any:
        pass

    @abstractmethod
    def parameters(self) -> list[Parameter]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def modules(self) -> list[nn.Module]:
        pass

    @abstractmethod
    def apply_to_module(self):
        pass

    @abstractmethod
    def extract_from_module(self, base_module: nn.Module):
        pass


class LoHaModule(PeftBase):
    w1d: nn.Module
    w1u: nn.Module
    w2d: nn.Module
    w2u: nn.Module
    alpha: Tensor
    dropout: Dropout

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float, layer_kwargs: dict = {}):
        super().__init__(prefix, orig_module, layer_kwargs)
        self.rank = rank
        self.alpha = torch.tensor(alpha)
        self.dropout = Dropout(0)

        if orig_module is not None:
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

        self.orig_forward = self.orig_module.forward if self.orig_module is not None else None

    def forward(self, x, *args, **kwargs):
        # They definitely exist at this point in the execution.
        assert self.op
        assert self.orig_module
        assert self.orig_forward

        train = self.orig_module.training
        ww1d = self.dropout(self.w1d) if train else self.w1d
        ww1u = self.dropout(self.w1u) if train else self.w1u
        ww2d = self.dropout(self.w2d) if train else self.w2d
        ww2u = self.dropout(self.w2u) if train else self.w2u
        W = custom_passes.LohaWeight.apply(ww1d, ww1u, ww2d, ww2u)
        return self.orig_forward(x) + \
            self.op(cast(Tensor, W), x, **self.layer_kwargs) * \
            (self.alpha / self.rank)


class LoRAModule(PeftBase):
    lora_down: nn.Module
    lora_up: nn.Module
    alpha: torch.Tensor
    dropout: Dropout

    # Note there's a few times in this class where we assert the existence of
    # optional members. This is because these members might not exist at
    # construction, but definitely exist by the time those methods are called.

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float, layer_kwargs: dict = {}):
        super(LoRAModule, self).__init__(prefix, orig_module, layer_kwargs)

        self.prefix = prefix.replace('.', '_')
        self.orig_module = orig_module
        self.rank = rank
        self.alpha = torch.tensor(alpha)
        self.dropout = Dropout(0)
        if orig_module is not None:
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

        self.orig_forward = self.orig_module.forward if self.orig_module is not None else None

    def initialize_weights(self):
        assert self.orig_module

        device = self.orig_module.weight.device
        match self.orig_module:
            case nn.Linear():
                in_features = self.orig_module.in_features
                out_features = self.orig_module.out_features
                self.lora_down = nn.Linear(in_features, self.rank, bias=False, device=device, **self.layer_kwargs)
                self.lora_up = nn.Linear(self.rank, out_features, bias=False, device=device, **self.layer_kwargs)

            case nn.Conv2d():
                in_channels = self.orig_module.in_channels
                out_channels = self.orig_module.out_channels
                self.lora_down = Conv2d(in_channels, self.rank, (1, 1), bias=False, device=device, **self.layer_kwargs)
                self.lora_up = Conv2d(self.rank, out_channels, (1, 1), bias=False, device=device, **self.layer_kwargs)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        # They definitely exist at this point in the execution.
        assert self.orig_module
        assert self.orig_forward

        if self.orig_module.training:
            ld = self.lora_up(self.dropout(self.lora_down(x)))
            return self.orig_forward(x) + ld * (self.alpha / self.rank)

        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

    def requires_grad_(self, requires_grad: bool):
        self.lora_down.requires_grad_(requires_grad)
        self.lora_up.requires_grad_(requires_grad)

    def to(self, device: torch.device|None = None, dtype: torch.dtype|None = None) -> 'LoRAModule':
        self.lora_down.to(device, dtype)
        self.lora_up.to(device, dtype)
        self.alpha.to(device, dtype)
        return self

    def parameters(self) -> list[Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())

    def load_state_dict(self, state_dict: dict):
        if self.prefix + ".lora_down.weight" in state_dict:
            down_state_dict = {
                "weight": state_dict.pop(self.prefix + ".lora_down.weight")
            }
            self.lora_down.load_state_dict(down_state_dict, strict=False)

        if self.prefix + ".lora_up.weight" in state_dict:
            up_state_dict = {
                "weight": state_dict.pop(self.prefix + ".lora_up.weight")
            }
            self.lora_up.load_state_dict(up_state_dict, strict=False)

        if self.prefix + ".alpha" in state_dict:
            self.alpha = state_dict.pop(self.prefix + ".alpha")

    def state_dict(self) -> dict:
        state_dict = {}
        state_dict[self.prefix + ".lora_down.weight"] = self.lora_down.weight.data
        state_dict[self.prefix + ".lora_up.weight"] = self.lora_up.weight.data
        state_dict[self.prefix + ".alpha"] = self.alpha
        return state_dict

    def modules(self) -> list[nn.Module]:
        return [self.lora_down, self.lora_up, self.dropout]

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


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

    def modules(self) -> list[nn.Module]:
        return []

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

    lora_modules: dict[str, LoRAModule]

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

        self.lora_modules = self.__create_modules(orig_module, alpha)

    def __create_modules(self, orig_module: nn.Module | None, alpha: float) -> dict[str, LoRAModule]:
        lora_modules = {}

        if orig_module is not None:
            for name, child_module in orig_module.named_modules():
                if len(self.module_filter) == 0 or any([x in name for x in self.module_filter]):
                    if isinstance(child_module, Linear) or \
                       isinstance(child_module, Conv2d):
                        lora_modules[name] = LoRAModule(self.prefix + "_" + name, child_module, self.rank, alpha)

        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for name, module in self.lora_modules.items():
            module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for name, module in self.lora_modules.items():
            parameters += module.parameters()
        return parameters

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModuleWrapper':
        for name, module in self.lora_modules.items():
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

        for name, module in self.lora_modules.items():
            module.load_state_dict(state_dict)

        # create dummy modules for the remaining keys
        remaining_names = list(state_dict.keys())
        for name in remaining_names:
            if name.endswith(".alpha"):
                prefix = name.removesuffix(".alpha")
                module = DummyLoRAModule(prefix)
                module.load_state_dict(state_dict)
                self.lora_modules[prefix] = module

    def state_dict(self) -> dict:
        """
        Returns the state dict
        """
        state_dict = {}

        for name, module in self.lora_modules.items():
            state_dict |= module.state_dict()

        return state_dict

    def modules(self) -> list[nn.Module]:
        """
        Returns a list of all modules
        """
        modules = []
        for module in self.lora_modules.values():
            modules += module.modules()

        return modules

    def hook_to_module(self):
        """
        Hooks the LoRA into the module without changing its weights
        """
        for name, module in self.lora_modules.items():
            module.hook_to_module()

    def remove_hook_from_module(self):
        """
        Removes the LoRA hook from the module without changing its weights
        """
        for name, module in self.lora_modules.items():
            module.remove_hook_from_module()

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for name, module in self.lora_modules.items():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for name, module in self.lora_modules.items():
            module.extract_from_module(base_module)

    def prune(self):
        """
        Removes all dummy modules
        """
        self.lora_modules = {k: v for (k, v) in self.lora_modules.items() if not isinstance(v, DummyLoRAModule)}

    def set_dropout(self, dropout_probability: float):
        """
        Sets the dropout probability
        """
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError("Dropout probability must be in [0, 1]")
        for module in self.lora_modules.values():
            module.dropout.p = dropout_probability
