import math
import abc
import typing
import inspect
import functools
import dataclasses
from modules.util.enum.LoraType import LoraType

from torch import nn, Tensor
from torch.nn import Dropout, Linear, Conv2d

@dataclasses.dataclass
class LoraOptions:
    rank: int
    alpha: float = 1.0
    dropout: float = 0.0
    type: LoraType = LoraType.LierLa


class BaseLoraModule(nn.Module, abc.ABC):
    lora_type: typing.ClassVar[LoraType]
    module_type: typing.ClassVar[typing.Type[nn.Module]]
    orig_forward: typing.Callable[[Tensor], Tensor] | None
    
    def __init__(self, orig_module: nn.Module | None) -> None:
        super().__init__()
        assert isinstance(orig_module, self.module_type)
        self.orig_forward = None if orig_module is None else lambda x: orig_module.forward(x)
    
    @property
    def is_dummy(self) -> bool:
        return self.orig_forward is None
    
    def __init_subclass__(cls: typing.Type["BaseLoraModule"]) -> None:
        super().__init_subclass__()
        if inspect.isabstract(cls):
            return
        
        assert (cls.lora_type, cls.module_type) not in MODULE_MAP
        MODULE_MAP[cls.lora_type, cls.module_type] = cls
        
MODULE_MAP = typing.Dict[tuple[LoraType, typing.Type[nn.Module]], typing.Type[BaseLoraModule]]

class LierlaLoraModule(BaseLoraModule, abc.ABC):
    lora_down: nn.Module
    lora_up: nn.Module
    alpha: Tensor
    dropout: typing.Callable[[Tensor], Tensor]

    def __init__(self, orig_module: nn.Module | None, options: LoraOptions):
        super().__init__(self.orig_module)
        self.rank = options.rank
        self.register_buffer("alpha", Tensor(options.alpha, device=None if self.is_dummy else orig_module.weight.device), persistent=True)
        self.dropout = lambda x: Dropout(options.dropout).forward(x)
        if not self.is_dummy:
            self.alpha = self.alpha.to(orig_module.weight.device)

    def forward(self, x, *args, **kwargs):
        if self.training:
            ld = self.lora_up(self.dropout(self.lora_down(x)))
            return self.orig_forward(x) + ld * (self.alpha / self.rank)

        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

    @functools.wraps(nn.Module.to)
    def to(self, *args, **kwargs) -> 'LierlaLoraModule':
        out = super().to(*args, **kwargs)
        self.alpha = self.alpha.to(*args, **kwargs)
        return out

    def apply_to_module(self):
        # TODO
        raise NotImplementedError

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        raise NotImplementedError


class LinearLierlaLoraModule(LierlaLoraModule):
    lora_type = LoraType.LierLa
    module_type = Linear
    
    def __init__(self, orig_module: Linear | None, options: LoraOptions):
        super().__init__(orig_module, options)

        in_features = orig_module.in_features
        out_features = orig_module.out_features

        self.lora_down = Linear(in_features, options.rank, bias=False, device=orig_module.weight.device)
        self.lora_up = Linear(options.rank, out_features, bias=False, device=orig_module.weight.device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class Conv2dLierlaLoraModule(LierlaLoraModule):
    lora_type = LoraType.LierLa
    module_type = Conv2d
    
    def __init__(self, orig_module: Conv2d | None, options: LoraOptions):
        super(Conv2dLierlaLoraModule, self).__init__(orig_module, options)
        in_channels = orig_module.in_channels
        out_channels = orig_module.out_channels

        self.lora_down = Conv2d(in_channels, options.rank, (1, 1), bias=False, device=orig_module.weight.device)
        self.lora_up = Conv2d(options.rank, out_channels, (1, 1), bias=False, device=orig_module.weight.device)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


class DiffusionLoraModule(nn.Module):
    def __init__(
            self,
            orig_module: nn.Module | None,
            options: LoraOptions,
            module_filter: list[str] | None = None,
    ):
        super().__init__()
        self.options = options
        self.__create_modules(orig_module, module_filter)

    def __create_modules(self, orig_module: nn.Module | None, module_filter: list[str] | None = None) -> None:
        if orig_module is not None:
            for name, child_module in orig_module.named_modules():
                attrname = name.replace(".", "_")
                if module_filter is None or any([x in name for x in self.module_filter]):
                    if isinstance(child_module, Linear):
                        setattr(self, attrname, MODULE_MAP[self.options.type, Linear](child_module, self.options))
                    elif isinstance(child_module, Conv2d):
                        setattr(self, attrname, MODULE_MAP[self.options.type, Conv2d](child_module, self.options))

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for module in self.children():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for module in self.children():
            module.extract_from_module(base_module)

