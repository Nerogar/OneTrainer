import copy
import math
from abc import abstractmethod
from typing import Any, Mapping, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Dropout, Linear, Conv2d, Parameter

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType


class PeftBase(nn.Module):
    is_applied: bool
    orig_forward: Any | None
    _orig_module: list[nn.Module] | None  # list prevents it from registering
    prefix: str
    layer_kwargs: dict  # Applied during the forward op() call.

    def __init__(self, prefix: str, orig_module: nn.Module | None):
        super().__init__()
        self.prefix = prefix.replace('.', '_') + '.'
        self._orig_module = [orig_module] if orig_module else None
        self.is_applied = False
        self.layer_kwargs = {}

        if orig_module is not None:
            match orig_module:
                case nn.Linear():
                    self.op = F.linear
                    self.shape = orig_module.weight.shape
                case nn.Conv2d():
                    self.op = F.conv2d
                    self.shape = orig_module.weight.shape
                    self.layer_kwargs.setdefault("stride", orig_module.stride)
                    self.layer_kwargs.setdefault("padding", orig_module.padding)
                    self.layer_kwargs.setdefault("dilation", orig_module.dilation)
                    self.layer_kwargs.setdefault("groups", orig_module.groups)
                case _:
                    raise NotImplementedError("Only Linear and Conv2d are supported layers.")

    def hook_to_module(self):
        if not self.is_applied:
            self.orig_forward = self.orig_module.forward
            self.orig_module.forward = self.forward
            self.is_applied = True

    def remove_hook_from_module(self):
        assert self.orig_forward
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.is_applied = False

    def make_weight(self, A: Tensor, B: Tensor):
        """Layer-type-independent way of creating a weight matrix from LoRA A/B.

        While linear layer types are a straightforward matrix multiplication of
        the weights, convolution is a little less straightforward. This function
        will take a PEFT A/B matrix and return the full-sized weight matrix.

        A should be the equivalent of "LoRA Down" in most code, and likewise B
        the equivalent of "LoRA Up".

        Thanks to KohakuBlueLeaf (author of LyCORIS) for showing me how to do
        this in a layer-independent fashion. I was tearing my hair out over
        wrangling the matrix shapes in a functionally correct manner before.
        """
        W = B.view(B.size(0), -1) @ A.view(A.size(0), -1)
        return W.view(self.shape)

    @property
    def orig_module(self) -> nn.Module:
        assert self._orig_module
        return self._orig_module[0]

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        state_dict = {k.removeprefix(self.prefix): v for (k, v) in state_dict.items() if k.startswith(self.prefix)}
        return super().load_state_dict(state_dict, strict, assign)

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def apply_to_module(self):
        pass

    @abstractmethod
    def extract_from_module(self, base_module: nn.Module):
        pass

    def create_layer(self) -> Tuple[nn.Module, nn.Module]:
        """Generic helper function for creating a PEFT layer, like LoRA.

        Creates down/up layer modules for the given layer type in the
        orig_module, for the given rank.

        Does not perform initialization, as that usually depends on the PEFT
        method.
        """
        device = self.orig_module.weight.device
        lora_down = None
        lora_up = None
        match self.orig_module:
            case nn.Linear():
                in_features = self.orig_module.in_features
                out_features = self.orig_module.out_features
                lora_down = nn.Linear(in_features, self.rank, bias=False, device=device)
                lora_up = nn.Linear(self.rank, out_features, bias=False, device=device)

            case nn.Conv2d():
                in_channels = self.orig_module.in_channels
                out_channels = self.orig_module.out_channels
                kernel_size = self.orig_module.kernel_size
                stride = self.orig_module.stride
                padding = self.orig_module.padding
                dilation = self.orig_module.dilation
                groups = self.orig_module.groups
                lora_down = Conv2d(in_channels, self.rank, kernel_size, stride, padding, dilation=dilation, bias=False, device=device)
                # Note: small departure here from part of the community.
                # The original Mcrosoft repo does it this way. The cloneofsimo
                # repo handles the groups in lora_down. We follow the Microsoft
                # way. In reality, there shouldn't be any difference.
                lora_up = Conv2d(self.rank, out_channels // groups, (1, 1), 1, bias=False, device=device)

            case _:
                    raise NotImplementedError("Only Linear and Conv2d are supported layers.")

        return lora_down, lora_up

    @classmethod
    def make_dummy(cls):
        """Create a dummy version of a PEFT class.

        Acts identically to one of the above regular PEFT modules, but doesn't
        actually train or hook to the module at all. Generally used to hold
        extra keys that aren't specified in the training configuration.
        """
        class Dummy(cls):
            def forward(self, *args, **kwargs):
                assert self.orig_module
                return PeftBase.forward(self, *args, **kwargs)

            def load_state_dict(self, state_dict: Mapping[str, Any],
                                strict: bool = True, assign: bool = False):
                self._state_dict = copy.deepcopy(state_dict)
                return nn.modules.module._IncompatibleKeys([], [])

            def state_dict(self, *args, **kwargs):  # type: ignore
                return self._state_dict

            def hook_to_module(self):
                pass

            def remove_hook_from_module(self):
                pass

            def apply_to_module(self):
                pass

            def extract_from_module(self, base_module: nn.Module):
                pass

        return Dummy


class LoHaModule(PeftBase):
    """Implementation of LoHa from Lycoris.

    Does not support Tucker decomposition, extra scalar, or weight decomposition
    (DoRA). Those could be supported eventually, but currently no burning demand
    and it was tough to do them in a sufficiently generic fashion.
    """

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super().__init__(prefix, orig_module)
        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))

        if orig_module is not None:
            self.initialize_weights()
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

    def initialize_weights(self):
        self.hada_w1_a, self.hada_w1_b = self.create_layer()
        self.hada_w2_a, self.hada_w2_b = self.create_layer()

        nn.init.normal_(self.hada_w1_a.weight, std=0.1)
        nn.init.normal_(self.hada_w1_b.weight, std=1)
        nn.init.constant_(self.hada_w2_a.weight, 0)
        nn.init.normal_(self.hada_w2_b.weight, std=1)

    def forward(self, x, *args, **kwargs):
        # They definitely exist at this point in the execution.
        assert self.op
        assert self.orig_module
        assert self.orig_forward

        W1 = self.make_weight(self.dropout(self.hada_w1_a.weight),
                              self.dropout(self.hada_w1_b.weight))
        W2 = self.make_weight(self.dropout(self.hada_w2_a.weight),
                              self.dropout(self.hada_w2_b.weight))
        W = (W1 * W2) * (self.alpha / self.rank)
        return self.orig_forward(x) + self.op(x, W, bias=None, **self.layer_kwargs)

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


class LoRAModule(PeftBase):
    lora_down: nn.Module
    lora_up: nn.Module
    rank: int
    alpha: torch.Tensor
    dropout: Dropout

    # Note there's a few times in this class where we assert the existence of
    # optional members. This is because these members might not exist at
    # construction, but definitely exist by the time those methods are called.

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super(LoRAModule, self).__init__(prefix, orig_module)

        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))

        if orig_module is not None:
            self.initialize_weights()
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

    def initialize_weights(self):
        self.lora_down, self.lora_up = self.create_layer()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        # They definitely exist at this point in the execution.
        assert self.orig_forward

        if self.orig_module.training:
            ld = self.lora_up(self.dropout(self.lora_down(x)))
            return self.orig_forward(x) + ld * (self.alpha / self.rank)

        return self.orig_forward(x) + self.lora_up(self.lora_down(x)) * (self.alpha / self.rank)

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


class DoRAModule(LoRAModule):
    """Weight-decomposed low rank adaptation.

    Not unlike LoRA in theory but the forward pass is significantly more
    complicated, as it involves taking the norm of the directional result.
    """
    dora_num_dims: int
    dora_scale: nn.Parameter

    def initialize_weights(self):
        super().initialize_weights()

        # Thanks to KohakuBlueLeaf once again for figuring out the shape
        # wrangling that works for both Linear and Convolutional layers. If you
        # were just doing this for Linear, it would be substantially simpler.
        orig_weight = self.orig_module.weight.detach().float()
        self.dora_num_dims = orig_weight.dim() - 1
        self.dora_scale = nn.Parameter(
            torch.norm(
                orig_weight.transpose(1, 0).reshape(orig_weight.shape[1], -1),
                dim=1, keepdim=True)
            .reshape(orig_weight.shape[1], *[1] * self.dora_num_dims)
            .transpose(1, 0)
            .to(dtype=self.orig_module.weight.dtype,
                device=self.orig_module.weight.device)
        )

    def forward(self, x, *args, **kwargs):
        # They definitely exist at this point in the execution.
        assert self.orig_forward

        A = self.lora_down.weight
        B = self.lora_up.weight
        WP = self.orig_module.weight + (self.make_weight(A, B) * (self.alpha / self.rank))
        # A norm should never really end up zero at any point, but epsilon just
        # to be safe if we underflow or something. Also, as per section 4.3 of
        # the paper, we treat the norm as a constant for the purposes of
        # backpropagation in order to save VRAM (to do this, we detach it from
        # the gradient graph).
        norm = WP.detach() \
                 .transpose(0, 1) \
                 .reshape(WP.shape[1], -1) \
                 .norm(dim=1, keepdim=True) \
                 .reshape(WP.shape[1], *[1] * self.dora_num_dims) \
                 .transpose(0, 1) + torch.finfo(WP.dtype).eps
        WP = self.dora_scale * (WP / norm)
        # In the DoRA codebase (and thus the paper results), they perform
        # dropout on the *input*, rather than between layers, so we duplicate
        # that here.
        return self.op(self.dropout(x),
                       WP,
                       self.orig_module.bias,
                       **self.layer_kwargs)


DummyLoRAModule = LoRAModule.make_dummy()
DummyDoRAModule = DoRAModule.make_dummy()
DummyLoHaModule = LoHaModule.make_dummy()


class LoRAModuleWrapper:
    orig_module: nn.Module
    rank: int
    alpha: float

    lora_modules: dict[str, LoRAModule]

    def __init__(
            self,
            orig_module: nn.Module | None,
            prefix: str,
            config: TrainConfig,
            module_filter: list[str] = None,
    ):
        self.orig_module = orig_module
        self.prefix = prefix
        self.peft_type = config.peft_type
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.module_filter = module_filter if module_filter is not None else []
        weight_decompose = config.lora_decompose
        if self.peft_type == PeftType.LORA:
            if weight_decompose:
                self.klass = DoRAModule
                self.dummy_klass = DummyDoRAModule
                self.additional_args = [self.rank, self.alpha]
            else:
                self.klass = LoRAModule
                self.dummy_klass = DummyLoRAModule
                self.additional_args = [self.rank, self.alpha]
        elif self.peft_type == PeftType.LOHA:
            self.klass = LoHaModule
            self.dummy_klass = DummyLoHaModule
            self.additional_args = [self.rank, self.alpha]

        self.lora_modules = self.__create_modules(orig_module)

    def __create_modules(self, orig_module: nn.Module | None) -> dict[str, LoRAModule]:
        lora_modules = {}

        if orig_module is not None:
            for name, child_module in orig_module.named_modules():
                if len(self.module_filter) == 0 or any([x in name for x in self.module_filter]):
                    if isinstance(child_module, Linear) or \
                       isinstance(child_module, Conv2d):
                        lora_modules[name] = self.klass(self.prefix + "_" + name, child_module, *self.additional_args)

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
            try:
                module.load_state_dict(state_dict)
            except RuntimeError:
                print(f"Missing key for {name}; initializing it to zero.")

        # Temporarily re-create the state dict, so we can see what keys were left.
        remaining_names = set(state_dict) - set(self.state_dict())

        # create dummy modules for the remaining keys
        for name in remaining_names:
            if name.endswith(".alpha"):
                prefix = name.removesuffix(".alpha")
                module = self.dummy_klass(prefix, None, *self.additional_args)
                module.load_state_dict(state_dict)
                self.lora_modules[prefix] = module

    def state_dict(self) -> dict:
        """
        Returns the state dict
        """
        state_dict = {}

        for name, module in self.lora_modules.items():
            state_dict |= module.state_dict(prefix=module.prefix)

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
        self.lora_modules = {k: v for (k, v) in self.lora_modules.items() if not isinstance(v, self.dummy_klass)}

    def set_dropout(self, dropout_probability: float):
        """
        Sets the dropout probability
        """
        if dropout_probability < 0 or dropout_probability > 1:
            raise ValueError("Dropout probability must be in [0, 1]")
        for module in self.lora_modules.values():
            module.dropout.p = dropout_probability
