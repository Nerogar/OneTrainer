import copy
import inspect
from collections.abc import Callable
from typing import Any

from modules.util.compile_util import init_compile
from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.torch_util import add_dummy_grad_fn_, has_grad_fn

import torch
from torch import nn

from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoIndividualTokenRefinerBlock,
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformerBlock,
)
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
from transformers.models.t5.modeling_t5 import T5Block

init_compile()


def _kwargs_to_args(fun: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    signature = dict(inspect.signature(fun).parameters)
    parameters = []

    for i, (key, value) in enumerate(signature.items()):
        if i < len(args):
            parameters.append(args[i])
        elif key in kwargs:
            parameters.append(kwargs[key])
        elif value.default is not value.empty:
            parameters.append(value.default)

    return tuple(parameters)


def __get_args_indices(fun: Callable, arg_names: list[str]) -> list[int]:
    signature = dict(inspect.signature(fun).parameters)
    indices = []

    for i, key in enumerate(signature.keys()):
        if key in arg_names:
            indices.append(i)

    return indices


__current_call_index = 0


def _generate_call_index() -> int:
    global __current_call_index
    __current_call_index += 1
    return __current_call_index


class BaseCheckpointLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CheckpointLayer(BaseCheckpointLayer):
    def __init__(self, orig_module: nn.Module, orig_forward, train_device: torch.device, checkpointing: bool = True):
        super().__init__()

        assert (orig_module is None or orig_forward is None) and not (orig_module is None and orig_forward is None)
        self.checkpoint = orig_module
        self.orig_forward = orig_forward
        self.checkpointing = checkpointing

        # dummy tensor that requires grad is needed for checkpointing to work when training a LoRA
        self.dummy = torch.zeros((1,), device=train_device, requires_grad=True)

    def __orig(self, *args, **kwargs):
        return self.orig_forward(*args, **kwargs) if self.checkpoint is None else self.checkpoint(*args, **kwargs)

    def __checkpointing_forward(self, dummy: torch.Tensor, *args, **kwargs):
        return self.__orig(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.__checkpointing_forward,
                self.dummy,
                *args,
                **kwargs,
                use_reentrant=False
            )
        else:
            return self.__orig(*args, **kwargs)

class OffloadCheckpointLayer(BaseCheckpointLayer):
    def __init__(self, orig_module: nn.Module, orig_forward, train_device: torch.device, conductor: LayerOffloadConductor, layer_index: int):
        super().__init__()

        assert (orig_module is None or orig_forward is None) and not (orig_module is None and orig_forward is None)
        self.checkpoint = orig_module
        self.orig_forward = orig_forward

        self.dummy = torch.zeros((1,), device=train_device, requires_grad=True)
        self.conductor = conductor
        self.layer_index = layer_index

    def __deepcopy__(self, memo):
        # conductor holds torch.cuda.Stream/Event objects that cannot be deep-copied or pickled.
        # deepcopy is only used at save time to build a dtype-converted CPU copy of the pipeline,
        # where the conductor is never invoked, so share the existing instance instead of copying it.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            result.__dict__[key] = value if key == "conductor" else copy.deepcopy(value, memo)
        return result

    def __checkpointing_forward(self, dummy: torch.Tensor, call_id: int, *args):
        init_compile()  # workaround for https://github.com/pytorch/pytorch/issues/186537
        if self.layer_index == 0 and not torch.is_grad_enabled():
            self.conductor.start_forward(True)

        args = self.conductor.before_layer(self.layer_index, call_id, args)
        output = self.orig_forward(*args) if self.checkpoint is None else self.checkpoint(*args)

        self.conductor.after_layer(self.layer_index, call_id, args)

        # make sure at least one of the output tensors has a grad_fn so the output of the checkpoint has a grad_fn.
        # this can only happen if a checkpointed block has no trainable parameters, because of a layer filter
        # was used. Adding a dummy grad function is a workaround required by use_reentrant==True checkpointing:
        if torch.is_grad_enabled() and not has_grad_fn(output):
            output = add_dummy_grad_fn_(output)

        return output

    def forward(self, *args, **kwargs):
        call_id = _generate_call_index()
        args = _kwargs_to_args(self.orig_forward if self.checkpoint is None else self.checkpoint.forward, args, kwargs)
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.__checkpointing_forward,
                self.dummy,
                call_id,
                *args,
                use_reentrant=True
            )
        else:
            if self.layer_index == 0:
                self.conductor.start_forward(False)

            args = self.conductor.before_layer(self.layer_index, call_id, args)
            output = self.orig_forward(*args) if self.checkpoint is None else self.checkpoint(*args)
            self.conductor.after_layer(self.layer_index, call_id, args)
            return output

def create_checkpoint(
        orig_module: nn.Module,
        train_device: torch.device,
        include_from_offload_param_names: list[str] = None,
        conductor: LayerOffloadConductor | None = None,
        checkpointing: bool = True,
        layer_index: int = 0,
        compile: bool = False,
) -> Callable:
    if include_from_offload_param_names is None:
        include_from_offload_param_names = []
    included_offload_param_indices = __get_args_indices(orig_module.forward, include_from_offload_param_names)

    if conductor is not None:
        conductor.add_layer(orig_module, included_offload_param_indices)

    if conductor is not None and conductor.offload_activated():
        # offloading is structurally coupled to use_reentrant=True checkpointing during the back pass:
        # the recompute is the only thing firing before_layer/after_layer in the backward direction, so
        # both layer and activation offloading need checkpointing to move tensors back for backward.
        # Rather than silently forcing checkpointing on when the part disabled it, reject the combination.
        if not checkpointing:
            raise NotImplementedError("offloading currently requires gradient checkpointing")
        if compile:
            layer = OffloadCheckpointLayer(orig_module=orig_module, orig_forward=None, train_device=train_device, conductor=conductor, layer_index=layer_index)
            #don't compile the checkpointing layer - offloading cannot be compiled:
            orig_module.compile(fullgraph=True)
            return layer
        else:
            #only patch forward() if possible. Inserting layers is necessary for torch.compile, but causes issues with at least 1 text encoder model. we don't compile text encoders
            layer = OffloadCheckpointLayer(orig_module=None, orig_forward=orig_module.forward, train_device=train_device, conductor=conductor, layer_index=layer_index)
            orig_module.forward = layer.forward
            return orig_module
    else:
        if compile:
            layer = CheckpointLayer(orig_module=orig_module, orig_forward=None, train_device=train_device, checkpointing=checkpointing)
            #do compile the checkpointing layer - slightly faster
            layer.compile(fullgraph=True)
            return layer
        else:
            layer = CheckpointLayer(orig_module=None, orig_forward=orig_module.forward, train_device=train_device, checkpointing=checkpointing)
            orig_module.forward = layer.forward
            return orig_module

def _create_checkpoints_for_module_list(
        module_list: nn.ModuleList,
        include_from_offload_param_names: list[str],
        conductor: LayerOffloadConductor,
        checkpointing: bool,
        train_device: torch.device,
        layer_index: int,
        compile: bool,
) -> int:

    for i, layer in enumerate(module_list):
        if isinstance(module_list[i], BaseCheckpointLayer):
            continue
        module_list[i] = create_checkpoint(
                layer, train_device,
                include_from_offload_param_names,
                conductor, checkpointing, layer_index, compile=compile,
            )
        layer_index += 1
    return layer_index

def _remove_checkpoint_keys(module, state_dict, prefix, local_metadata):
    for k in list(state_dict.keys()):
        if ".checkpoint." in k:
            state_dict[k.replace(".checkpoint.", ".")] = state_dict.pop(k)

def enable_checkpointing(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
        compile: bool,
        lists, # if there are multiple entries in this list, they must be in the exact order they are executed - otherwise offloading fails
        offload_enabled: bool = True,
) -> LayerOffloadConductor | None:
    if not part.checkpointing_or_offloading_enabled() and not compile:
        return None

    # a conductor exists iff this part actually offloads (and the component supports conductor offloading)
    offload = offload_enabled and part.offloading_enabled()
    conductor = LayerOffloadConductor(model, config, part) if offload else None
    checkpointing = part.checkpointing_enabled()

    layer_index = 0
    for type_or_list, param_names in lists:

        assert isinstance(type_or_list, (nn.ModuleList, type))
        if isinstance(type_or_list, nn.ModuleList):
            module_list = type_or_list
            layer_index = _create_checkpoints_for_module_list(
                module_list,
                param_names,
                conductor,
                checkpointing,
                torch.device(config.train_device),
                layer_index,
                compile = compile,
            )
        else:
            t = type_or_list
            for child_module in model.modules():
                if isinstance(child_module, nn.ModuleList) and isinstance(child_module[0], t):
                    module_list = child_module
                    assert all(isinstance(m, t) for m in child_module)
                    layer_index = _create_checkpoints_for_module_list(
                        module_list,
                        param_names,
                        conductor,
                        checkpointing,
                        torch.device(config.train_device),
                        layer_index,
                        compile = compile,
                    )
    model._register_state_dict_hook(_remove_checkpoint_keys)
    return conductor

def enable_checkpointing_for_basic_transformer_blocks(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
        offload_enabled: bool,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
            (BasicTransformerBlock  ,        []),
        ],
        offload_enabled = offload_enabled,
    )

def enable_checkpointing_for_clip_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
):
    return enable_checkpointing(model, config, part, False, [
        (CLIPEncoderLayer, []), # No activation offloading for text encoders, because the output might be taken from the middle of the network
    ], offload_enabled=False) # CLIP is non-offloadable; keep it plain-checkpointed so a migrated offload_fraction can't build a self-activating conductor

def enable_checkpointing_for_t5_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (T5Block, []),
    ])


def enable_checkpointing_for_gemma_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (Gemma2DecoderLayer, []),
    ])


def enable_checkpointing_for_llama_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (LlamaDecoderLayer, []),
    ])

def enable_checkpointing_for_mistral_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (MistralDecoderLayer, []),  # no activation offloading: this encoder is never trained
    ])



def enable_checkpointing_for_qwen25vl_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (Qwen2_5_VLDecoderLayer, []),  # TODO No activation offloading for other encoders, see above. But clip skip is not implemented for QwenVL. Then do activation offloading?
    ])

def enable_checkpointing_for_qwen3_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (Qwen3DecoderLayer, []),  # no activation offloading: this encoder is never trained
    ])

def enable_checkpointing_for_stable_diffusion_3_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (JointTransformerBlock, ["hidden_states", "encoder_hidden_states"]),
    ])

def enable_checkpointing_for_flux_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"                         ]),
    ])

def enable_checkpointing_for_flux2_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"                         ]),
    ])


def enable_checkpointing_for_chroma_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.transformer_blocks,        ["hidden_states", "encoder_hidden_states"]),
        (model.single_transformer_blocks, ["hidden_states"                         ]),
    ])


def enable_checkpointing_for_qwen_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.transformer_blocks, ["hidden_states", "encoder_hidden_states"]),
    ])

def enable_checkpointing_for_z_image_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.noise_refiner, ["x"]),
        (model.context_refiner, ["x"]),
        (model.layers, ["x"]),
    ])


def enable_checkpointing_for_sana_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (SanaTransformerBlock, ["hidden_states"]),
    ])

def enable_checkpointing_for_hunyuan_video_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (HunyuanVideoIndividualTokenRefinerBlock, ["hidden_states"                         ]),
        (HunyuanVideoTransformerBlock,            ["hidden_states", "encoder_hidden_states"]),
        (HunyuanVideoSingleTransformerBlock,      ["hidden_states"                         ]),
    ])

def enable_checkpointing_for_hi_dream_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.double_stream_blocks, ["hidden_states", "encoder_hidden_states"]),
        (model.single_stream_blocks, ["hidden_states"                         ]),
    ])

def enable_checkpointing_for_ernie_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.layers, ["x"]),
    ])

def enable_checkpointing_for_krea2_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    # Krea2TransformerBlock takes (hidden_states, temb, image_rotary_emb, attention_mask).
    return enable_checkpointing(model, config, part, config.compile, [
        (model.text_fusion.layerwise_blocks, ["hidden_states"]),
        (model.text_fusion.refiner_blocks,   ["hidden_states"]),
        (model.transformer_blocks,           ["hidden_states"]),
    ])

def enable_checkpointing_for_qwen3vl_encoder_layers(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, False, [
        (Qwen3VLTextDecoderLayer, []),  # no activation offloading: this encoder is never trained
    ])
