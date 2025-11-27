import copy
import math
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from modules.module.oft_utils import OFTRotationModule
from modules.module.quantized.LinearSVD import BaseLinearSVD
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType
from modules.util.ModuleFilter import ModuleFilter
from modules.util.quantization_util import get_unquantized_weight, get_weight_shape

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Conv2d, Dropout, Linear, Parameter


class PeftBase(nn.Module):
    is_applied: bool
    orig_forward: Any | None
    orig_eval: Any | None
    orig_train: Any | None
    _orig_module: list[nn.Module] | None  # list prevents it from registering
    prefix: str
    layer_kwargs: dict  # Applied during the forward op() call.
    _initialized: bool  # Tracks whether we've created the layers or not.

    def __init__(self, prefix: str, orig_module: nn.Module | None):
        super().__init__()
        self.prefix = prefix + '.'
        self._orig_module = [orig_module] if orig_module else None
        self.is_applied = False
        self.layer_kwargs = {}
        self._initialized = False

        if orig_module is not None:
            match orig_module:
                case nn.Linear():
                    self.op = F.linear
                    self.shape = get_weight_shape(orig_module)
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
            self.orig_train = self.orig_module.train
            self.orig_eval = self.orig_module.eval
            self.orig_module.forward = self.forward
            self.orig_module.train = self._wrap_train
            self.orig_module.eval = self._wrap_eval
            self.is_applied = True

    def remove_hook_from_module(self):
        assert self.orig_forward is not None
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.orig_module.train = self.orig_train
            self.orig_module.eval = self.orig_eval
            self.is_applied = False

    def _wrap_train(self, mode=True):
        self.orig_train(mode)
        self.train(mode)

    def _wrap_eval(self):
        self.orig_eval()
        self.eval()

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

    def check_initialized(self):
        """Checks, and raises an exception, if the module is not initialized."""
        if not self._initialized:
            raise RuntimeError(f"Module {self.prefix} is not initialized.")

        # Perform assertions to make pytype happy.
        assert self.orig_forward is not None
        assert self.orig_module is not None
        assert self.op is not None

    @property
    def orig_module(self) -> nn.Module:
        assert self._orig_module is not None
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

    def create_layer(self) -> tuple[nn.Module, nn.Module]:
        """Generic helper function for creating a PEFT layer, like LoRA.

        Creates down/up layer modules for the given layer type in the
        orig_module, for the given rank.

        Does not perform initialization, as that usually depends on the PEFT
        method.
        """
        device = self.orig_module.weight.device
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
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._state_dict = {}

            def forward(self, *args, **kwargs):
                assert self.orig_module is not None
                return PeftBase.forward(self, *args, **kwargs)

            def load_state_dict(self, state_dict: Mapping[str, Any],
                                strict: bool = True, assign: bool = False):
                self._initialized = True
                self._state_dict = copy.deepcopy(state_dict)
                # noinspection PyProtectedMember
                return nn.modules.module._IncompatibleKeys([], [])

            def state_dict(self, *args, **kwargs):  # type: ignore
                if not self._initialized:
                    raise RuntimeError("A state dict must be loaded before one can be returned.")
                return self._state_dict

            def initialize_weights(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def hook_to_module(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def remove_hook_from_module(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def apply_to_module(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def extract_from_module(self, base_module: nn.Module):
                raise NotImplementedError("Should never be called on a dummy module.")

        return Dummy


class LoHaModule(PeftBase):
    """Implementation of LoHa from Lycoris.

    Does not support Tucker decomposition, extra scalar, or weight decomposition
    (DoRA). Those could be supported eventually, but currently no burning demand
    and it was tough to do them in a sufficiently generic fashion.
    """

    rank: int
    dropout: Dropout
    hada_w1_a: Tensor | None
    hada_w1_b: Tensor | None
    hada_w2_a: Tensor | None
    hada_w2_b: Tensor | None

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super().__init__(prefix, orig_module)
        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.hada_w1_a = None
        self.hada_w1_b = None
        self.hada_w2_a = None
        self.hada_w2_b = None

        if orig_module is not None:
            self.initialize_weights()
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

    def initialize_weights(self):
        self._initialized = True

        hada_w1_b, hada_w1_a = self.create_layer()
        hada_w2_b, hada_w2_a = self.create_layer()
        self.hada_w1_a = hada_w1_a.weight
        self.hada_w1_b = hada_w1_b.weight
        self.hada_w2_a = hada_w2_a.weight
        self.hada_w2_b = hada_w2_b.weight

        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=1)
        nn.init.constant_(self.hada_w2_a, 0)
        nn.init.normal_(self.hada_w2_b, std=1)

    def check_initialized(self):
        super().check_initialized()
        assert self.hada_w1_a is not None
        assert self.hada_w1_b is not None
        assert self.hada_w2_a is not None
        assert self.hada_w2_b is not None

    def forward(self, x, *args, **kwargs):
        # They definitely exist at this point in the execution.
        self.check_initialized()

        # Yeah, yeah, it's different from the A/B parameters in make_weight.
        # Lycoris defines them in the opposite order. Yeah, it's confusing.
        W1 = self.make_weight(self.dropout(self.hada_w1_b),
                              self.dropout(self.hada_w1_a))
        W2 = self.make_weight(self.dropout(self.hada_w2_b),
                              self.dropout(self.hada_w2_a))
        W = (W1 * W2) * (self.alpha / self.rank)
        return self.orig_forward(x) + self.op(x, W, bias=None, **self.layer_kwargs)

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


class LoRAModule(PeftBase):
    lora_down: nn.Module | None
    lora_up: nn.Module | None
    rank: int
    alpha: torch.Tensor
    dropout: Dropout

    # Note there's a few times in this class where we assert the existence of
    # optional members. This is because these members might not exist at
    # construction, but definitely exist by the time those methods are called.

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super().__init__(prefix, orig_module)

        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.lora_down = None
        self.lora_up = None

        if orig_module is not None:
            self.initialize_weights()
            self.alpha = self.alpha.to(orig_module.weight.device)
        self.alpha.requires_grad_(False)

    def initialize_weights(self):
        self._initialized = True
        self.lora_down, self.lora_up = self.create_layer()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def check_initialized(self):
        super().check_initialized()
        assert self.lora_down is not None
        assert self.lora_up is not None

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if isinstance(self.orig_module, BaseLinearSVD):
            return self.orig_module.forward_with_lora(x, self.lora_down, self.lora_up, self.dropout, self.alpha)

        ld = self.lora_up(self.dropout(self.lora_down(x)))
        return self.orig_forward(x) + ld * (self.alpha / self.rank)

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass


class OFTModule(PeftBase):
    oft_R: OFTRotationModule | None
    rank: int
    oft_block_size: int
    coft: bool
    coft_eps: float
    block_share: bool
    dropout_probability: float
    adjustment_info: tuple[int, int] | None # for reporting

    def __init__(self, prefix: str, orig_module: nn.Module | None, oft_block_size: int, coft: bool, coft_eps: float, block_share: bool, **kwargs):
        super().__init__(prefix, orig_module)
        self.oft_block_size = oft_block_size
        self.rank = 0
        self.coft = coft
        self.coft_eps = coft_eps
        self.block_share = block_share
        self.dropout_probability = kwargs.pop('dropout_probability', 0.0)
        self.oft_R = None
        self.adjustment_info = None


        if orig_module is not None:
            self.initialize_weights()

    def adjust_oft_parameters(self, in_features, params):
        """
        Adjust the OFT parameters to be divisible by the in_features dimension.
        """
        if params < in_features:
            higher_params = params
            while higher_params <= in_features and in_features % higher_params != 0:
                higher_params += 1
        else:
            return in_features

        lower_params = params
        while lower_params > 1 and in_features % lower_params != 0:
            lower_params -= 1

        if (params - lower_params) <= (higher_params - params):
            return lower_params
        else:
            return higher_params

    def initialize_weights(self):
        self._initialized = True

        if isinstance(self.orig_module, nn.Linear):
            in_features = self.orig_module.in_features
        elif isinstance(self.orig_module, nn.Conv2d):
            if self.orig_module.dilation[0] > 1 or self.orig_module.dilation[1] > 1:
                raise ValueError("Conv2d with dilation > 1 is not supported by OFT.")
            in_features = self.orig_module.in_channels * self.orig_module.kernel_size[0] * self.orig_module.kernel_size[1]
        else:
            raise NotImplementedError("Unsupported layer type for OFT")

        oft_block_size = self.oft_block_size
        if oft_block_size <= 0:
            raise ValueError("Rank must be a positive.")

        # Adjust oft_block_size to be a divisor of in_features
        if in_features % oft_block_size != 0 or oft_block_size > in_features:
            old_oft_block_size = oft_block_size
            oft_block_size = self.adjust_oft_parameters(in_features, oft_block_size)
            self.adjustment_info = (old_oft_block_size, oft_block_size)

        # Calculate the number of blocks 'r'
        r = in_features // oft_block_size

        # Store the final, potentially adjusted values
        self.rank = r
        self.oft_block_size = oft_block_size

        n_elements = self.oft_block_size * (self.oft_block_size - 1) // 2

        self.oft_R = OFTRotationModule(
            r=self.rank if not self.block_share else 1,
            n_elements=n_elements,
            block_size=self.oft_block_size,
            in_features=in_features,
            coft=self.coft,
            coft_eps=self.coft_eps,
            block_share=self.block_share,
            use_cayley_neumann=True,
            num_cayley_neumann_terms=5,
            dropout_probability=self.dropout_probability,
        )

        nn.init.zeros_(self.oft_R.weight)

    def forward(self, x, *args, **kwargs):
        self.check_initialized()

        # For Linear layers, rotating the input is mathematically equivalent to rotating the weights.
        if isinstance(self.orig_module, nn.Linear):
            rotated_x = self.oft_R(x)
            return self.orig_forward(rotated_x, *args, **kwargs)

        # For Conv2d, we must rotate the weights, not the input, to preserve spatial information.
        orth_rotate = self.oft_R._cayley_batch(
            self.oft_R.weight, self.oft_R.block_size, self.oft_R.use_cayley_neumann, self.oft_R.num_cayley_neumann_terms
        )
        orth_rotate = self.oft_R.dropout(orth_rotate)

        if self.block_share:
            orth_rotate = orth_rotate.repeat(self.rank, 1, 1)

        weight = self.orig_module.weight
        weight_reshaped = weight.reshape(weight.shape[0], self.rank, self.oft_block_size)
        rotated_weight_reshaped = torch.einsum("ork,rkc->orc", weight_reshaped, orth_rotate)

        rotated_weight = rotated_weight_reshaped.reshape(weight.shape)

        return self.op(x, rotated_weight, self.orig_module.bias, **self.layer_kwargs)

    def apply_to_module(self):
        # TODO
        pass

    def extract_from_module(self, base_module: nn.Module):
        # TODO
        pass

    def check_initialized(self):
        super().check_initialized()
        assert self.oft_R is not None

    @property
    def dropout(self):
        return self.oft_R.dropout


class DoRAModule(LoRAModule):
    """Weight-decomposed low rank adaptation.

    Not unlike LoRA in theory but the forward pass is significantly more
    complicated, as it involves taking the norm of the directional result.
    """
    dora_num_dims: int
    dora_scale: Tensor | None
    norm_epsilon: bool
    decompose_output_axis: bool

    def __init__(self, *args, **kwargs):
        self.dora_scale = None
        self.norm_epsilon = kwargs.pop('norm_epsilon', False)
        self.decompose_output_axis = kwargs.pop('decompose_output_axis', False)
        self.train_device = kwargs.pop('train_device')
        super().__init__(*args, **kwargs)

    def initialize_weights(self):
        super().initialize_weights()

        if isinstance(self.orig_module, nn.Linear):
            orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
        else:
            assert isinstance(self.orig_module, nn.Conv2d)
            orig_weight = self.orig_module.weight.detach().float()

        # Thanks to KohakuBlueLeaf once again for figuring out the shape
        # wrangling that works for both Linear and Convolutional layers. If you
        # were just doing this for Linear, it would be substantially simpler.
        self.dora_num_dims = orig_weight.dim() - 1
        if self.decompose_output_axis:
            self.dora_scale = nn.Parameter(
                torch.norm(
                    orig_weight.reshape(orig_weight.shape[0], -1),
                    dim=1, keepdim=True)
                .reshape(orig_weight.shape[0], *[1] * self.dora_num_dims)
                .to(device=self.orig_module.weight.device)
            )
        else:
            self.dora_scale = nn.Parameter(
                torch.norm(
                    orig_weight.transpose(1, 0).reshape(orig_weight.shape[1], -1),
                    dim=1, keepdim=True)
                .reshape(orig_weight.shape[1], *[1] * self.dora_num_dims)
                .transpose(1, 0)
                .to(device=self.orig_module.weight.device)
            )

        del orig_weight

    def check_initialized(self):
        super().check_initialized()
        assert self.dora_scale is not None

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        A = self.lora_down.weight
        B = self.lora_up.weight

        if isinstance(self.orig_module, nn.Linear):
            orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
        else:
            assert isinstance(self.orig_module, nn.Conv2d)
            orig_weight = self.orig_module.weight.detach().float()

        WP = orig_weight + (self.make_weight(A, B) * (self.alpha / self.rank))
        del orig_weight
        # A norm should never really end up zero at any point, but epsilon just
        # to be safe if we underflow or something. Also, as per section 4.3 of
        # the paper, we treat the norm as a constant for the purposes of
        # backpropagation in order to save VRAM (to do this, we detach it from
        # the gradient graph).
        eps = torch.finfo(WP.dtype).eps if self.norm_epsilon else 0.0
        if self.decompose_output_axis:
            norm = WP.detach() \
                    .reshape(WP.shape[0], -1) \
                    .norm(dim=1) \
                    .reshape(WP.shape[0], *[1] * self.dora_num_dims) \
                    + eps
        else:
            norm = WP.detach() \
                    .transpose(0, 1) \
                    .reshape(WP.shape[1], -1) \
                    .norm(dim=1, keepdim=True) \
                    .reshape(WP.shape[1], *[1] * self.dora_num_dims) \
                    .transpose(0, 1) + eps
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
DummyOFTModule = OFTModule.make_dummy()


class LoRAModuleWrapper:
    orig_module: nn.Module
    rank: int
    alpha: float
    module_filters: list[ModuleFilter]

    lora_modules: dict[str, PeftBase]

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

        self.module_filters = [
            ModuleFilter(pattern, use_regex=config.layer_filter_regex)
            for pattern in (module_filter or [])
        ]

        weight_decompose = config.lora_decompose
        if self.peft_type == PeftType.LORA:
            if weight_decompose:
                self.klass = DoRAModule
                self.dummy_klass = DummyDoRAModule
                self.additional_args = [self.rank, self.alpha]
                self.additional_kwargs = {
                    'norm_epsilon': config.lora_decompose_norm_epsilon,
                    'decompose_output_axis': config.lora_decompose_output_axis,
                    'train_device': torch.device(config.train_device),
                }
            else:
                self.klass = LoRAModule
                self.dummy_klass = DummyLoRAModule
                self.additional_args = [self.rank, self.alpha]
                self.additional_kwargs = {}
        elif self.peft_type == PeftType.LOHA:
            self.klass = LoHaModule
            self.dummy_klass = DummyLoHaModule
            self.additional_args = [self.rank, self.alpha]
            self.additional_kwargs = {}
        elif self.peft_type == PeftType.OFT_2:
            self.klass = OFTModule
            self.dummy_klass = DummyOFTModule
            self.additional_args = [
                config.oft_block_size,
                config.oft_coft,
                config.coft_eps,
                config.oft_block_share,
            ]
            self.additional_kwargs = {
                'dropout_probability': config.dropout_probability,
            }

        self.lora_modules = self.__create_modules(orig_module, config)

    def __create_modules(self, orig_module: nn.Module | None, config: TrainConfig) -> dict[str, PeftBase]:
        if orig_module is None:
            return {}

        lora_modules = {}
        selected = []
        deselected = []
        unsuitable = []
        oft_adjustments = []

        for name, child_module in orig_module.named_modules():
            name = name.replace(".checkpoint.", ".")
            if not isinstance(child_module, Linear | Conv2d):
                unsuitable.append(name)
                continue
            if len(self.module_filters) == 0 or any(f.matches(name) for f in self.module_filters):
                lora_module = self.klass(self.prefix + "." + name, child_module, *self.additional_args, **self.additional_kwargs)
                lora_modules[name] = lora_module
                if self.peft_type == PeftType.OFT_2 and lora_module.adjustment_info:
                    old, new = lora_module.adjustment_info
                    oft_adjustments.append({'old': old, 'new': new})
                selected.append(name)
            else:
                deselected.append(name)

        if oft_adjustments:
            summary = defaultdict(int)
            for adj in oft_adjustments:
                summary[(adj['old'], adj['new'])] += 1

            sorted_summary = sorted(summary.items(), key=lambda item: (item[0][0], item[0][1]))

            summary_lines = [
                f"  - {count} layer{'s' if count > 1 else ''} from {old} to {new}"
                for (old, new), count in sorted_summary
            ]
            print(f"OFT Block Size automatically adjusted for {len(oft_adjustments)} layers. Changes:")
            print("\n".join(summary_lines))

        if len(self.module_filters) > 0:
            if config.debug_mode:
                print(f"Selected layers: {selected}")
                print(f"Deselected layers: {deselected}")
                print(f"Unsuitable for LoRA training: {unsuitable}")
            else:
                print(f"Selected layers: {len(selected)}")
                print(f"Deselected layers: {len(deselected)}")
                print("Note: Enable Debug mode to see the full list of layer names")

        unused_filters = [mf for mf in self.module_filters if not mf.was_used()]
        if len(unused_filters) > 0:
            raise ValueError('Custom layer filters: no modules were matched by the custom filter(s)')

        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for module in self.lora_modules.values():
            module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for module in self.lora_modules.values():
            parameters += module.parameters()
        return parameters

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModuleWrapper':
        for module in self.lora_modules.values():
            module.to(device, dtype)
        return self

    def _check_rank_matches(self, state_dict: dict[str, Tensor]):
        if not state_dict:
            return

        # For OFT, the comparison is not straightforward, so we skip it.
        if self.peft_type == PeftType.OFT_2:
            return

        if rank_key := next((k for k in state_dict if k.endswith((".lora_down.weight", ".hada_w1_a"))), None):
            if (checkpoint_rank := state_dict[rank_key].shape[0]) != self.rank:
                raise ValueError(f"Rank mismatch: checkpoint={checkpoint_rank}, config={self.rank}, please correct in the UI.")

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True):
        """
        Loads the state dict

        Args:
            state_dict: the state dict
            strict: whether to strictly enforce that the keys in state_dict match the module's parameters
        """
        # create a copy, so the modules can pop states
        state_dict = {k: v for (k, v) in state_dict.items() if k.startswith(self.prefix)}

        self._check_rank_matches(state_dict)

        try:
            for module in self.lora_modules.values():
                module.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            raise RuntimeError(f"Error during loading of module key \"{module.prefix}\"") from e

        # Temporarily re-create the state dict, so we can see what keys were left.
        remaining_names = set(state_dict) - set(self.state_dict())

        # create dummy modules for the remaining keys
        for name in remaining_names:
            if name.endswith(".alpha"):
                prefix = name.removesuffix(".alpha")
                module = self.dummy_klass(prefix, None, *self.additional_args, **self.additional_kwargs)
                module.load_state_dict(state_dict)
                self.lora_modules[prefix] = module

    def state_dict(self) -> dict:
        """
        Returns the state dict
        """
        state_dict = {}

        for module in self.lora_modules.values():
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
        for module in self.lora_modules.values():
            module.hook_to_module()

    def remove_hook_from_module(self):
        """
        Removes the LoRA hook from the module without changing its weights
        """
        for module in self.lora_modules.values():
            module.remove_hook_from_module()

    def apply_to_module(self):
        """
        Applys the LoRA to the module, changing its weights
        """
        for module in self.lora_modules.values():
            module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """
        Creates a LoRA from the difference between the base_module and the orig_module
        """
        for module in self.lora_modules.values():
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
