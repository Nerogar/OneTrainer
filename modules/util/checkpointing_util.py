import copy
import inspect
from collections.abc import Callable
from typing import Any

from modules.util.compile_util import init_compile
from modules.util.config.TrainConfig import TrainConfig, TrainModelPartConfig
from modules.util.LayerOffloadConductor import LayerOffloadConductor

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


def _view_key(tensor: torch.Tensor) -> tuple:
    # Identifies a tensor by the memory it reads: first-element address, extents, steps, element type.
    # Two tensors agreeing on all four are the same view of the same data, since live storages cannot
    # overlap. Used instead of identity because autograd hands back aliases, not the original objects.
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype


def __get_args_indices(fun: Callable, arg_names: list[str]) -> list[int]:
    signature = dict(inspect.signature(fun).parameters)
    indices = []

    for i, key in enumerate(signature.keys()):
        if key in arg_names:
            indices.append(i)

    return indices


class BaseCheckpointLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CheckpointLayer(BaseCheckpointLayer):
    def __init__(self, orig_module: nn.Module, orig_forward, checkpointing: bool = True):
        super().__init__()

        assert (orig_module is None or orig_forward is None) and not (orig_module is None and orig_forward is None)
        self.checkpoint = orig_module
        self.orig_forward = orig_forward
        self.checkpointing = checkpointing

    def __orig(self, *args, **kwargs):
        return self.orig_forward(*args, **kwargs) if self.checkpoint is None else self.checkpoint(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.__orig,
                *args,
                **kwargs,
                use_reentrant=False
            )
        else:
            return self.__orig(*args, **kwargs)


class LoadBoundary(torch.autograd.Function):
    # Identity op on a block's grad-carrying INPUT tensors, driving the conductor only: forward calls
    # before_layer, backward calls after_layer. Sitting on the input makes its backward fire LAST in
    # the block's backward.
    @staticmethod
    def forward(ctx, conductor, layer_index, *tensors):
        ctx.conductor = conductor
        ctx.layer_index = layer_index
        conductor.before_layer(layer_index, is_forward=True)
        return tensors
    @staticmethod
    def backward(ctx, *grads):
        # calling after_layer here, rather than after the forward, makes the train event cover the block's
        # backward kernels, so this layer's offload transfer cannot overlap them.
        ctx.conductor.after_layer(ctx.layer_index, list(grads))
        return (None, None, *grads)


class EvictBoundary(torch.autograd.Function):
    # Identity op on a block's grad-carrying OUTPUT tensors: forward calls after_layer, backward calls
    # before_layer. Sitting on the output makes its backward fire FIRST - before the checkpoint
    # rematerializes the block, so the weights are in by then.
    @staticmethod
    def forward(ctx, conductor, layer_index, *tensors):
        ctx.conductor = conductor
        ctx.layer_index = layer_index
        conductor.after_layer(layer_index, list(tensors))
        return tensors
    @staticmethod
    def backward(ctx, *grads):
        # workaround for https://github.com/pytorch/pytorch/issues/186537. This is the block's first
        # backward node, and it runs on the autograd worker thread - which is where AOTAutograd
        # compiles the backward graph, and which does not inherit the main thread's dynamo config.
        init_compile()
        ctx.conductor.before_layer(ctx.layer_index, is_forward=False)
        ctx.conductor.prefetch_activations(ctx.layer_index - 1)
        return (None, None, *grads)


def _apply_boundary(boundary, conductor, layer_index, values: tuple) -> tuple:
    # Wrap all grad-requiring tensors in `values` in a single boundary Function, so its backward fires
    # exactly once around the checkpoint's recompute and no gradient can reach the checkpoint without
    # crossing the boundary. Non-tensor and grad-free entries pass through untouched.
    indices = [i for i, v in enumerate(values) if isinstance(v, torch.Tensor) and v.requires_grad]
    if not indices:
        return values
    wrapped = boundary.apply(conductor, layer_index, *(values[i] for i in indices))
    values = list(values)
    for j, i in enumerate(indices):
        values[i] = wrapped[j]
    return tuple(values)


class BoundaryOffloadCheckpointLayer(BaseCheckpointLayer):
    # Weight movement driven by the LoadBoundary / EvictBoundary autograd Functions, which run eagerly
    # outside the compiled region. The block's forward and backward each stay a single traced graph, which
    # is what makes this path compatible with torch.compile's cudagraph trees. Activation offloading, when
    # enabled, rides on saved_tensors_hooks around the block.
    def __init__(self, orig_module: nn.Module, orig_forward, conductor: LayerOffloadConductor, layer_index: int, checkpointing: bool, included_offload_param_indices: list[int], compile: bool):
        super().__init__()

        assert (orig_module is None or orig_forward is None) and not (orig_module is None and orig_forward is None)
        self.checkpoint = orig_module
        self.orig_forward = orig_forward
        self.conductor = conductor
        self.layer_index = layer_index
        self.checkpointing = checkpointing
        self.included_offload_param_indices = included_offload_param_indices
        # compile the block together with its checkpoint, not the bare block: dynamo then traces the
        # checkpoint as a higher-order op and the min-cut partitioner prunes the recompute to what the
        # backward needs. With the checkpoint outside the compiled region the backward re-runs the whole
        # block. Traceable at all only because no conductor call happens in here.
        self.run_block = torch.compile(self.__checkpointed_block, fullgraph=True) if compile else self.__checkpointed_block

    def __deepcopy__(self, memo):
        # conductor holds torch.cuda.Stream/Event objects that cannot be deep-copied or pickled.
        # deepcopy is only used at save time to build a dtype-converted CPU copy of the pipeline,
        # where the conductor is never invoked, so share the existing instance instead of copying it.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # run_block is shared for the same reason as conductor: it closes over this instance, and
        # deepcopy is only used at save time where it is never invoked.
        for key, value in self.__dict__.items():
            result.__dict__[key] = value if key in ("conductor", "run_block") else copy.deepcopy(value, memo)
        return result

    def __orig(self, *args):
        return self.orig_forward(*args) if self.checkpoint is None else self.checkpoint(*args)

    def __checkpointed_block(self, *args):
        if self.checkpointing:
            # recompute the block in the backward, pruned by the min-cut partitioner when compiled
            return torch.utils.checkpoint.checkpoint(self.__orig, *args, use_reentrant=False)
        # no checkpointing: run once, autograd keeps every activation the backward needs
        return self.__orig(*args)

    def __run_block(self, args):
        def run():
            return self.run_block(*args)

        if not self.conductor.offloads_activations():
            return run()
        # Offload only the declared activation args. The declaration carries cross-block knowledge the
        # min-cut partitioner cannot have, seeing one block at a time: a tensor shared by every block
        # (rotary embeddings, masks) is saved once per block, and offloading those copies frees nothing
        # because the original stays live for the remaining blocks.
        #
        # Matched by view rather than by identity, because autograd saves an alias of the arg rather than
        # the arg object, so id() misses. Grad-agnostic: a LoRA layer filter can leave an offloaded
        # activation grad-free.
        layer_index = self.layer_index
        targets = {_view_key(args[i]) for i in self.included_offload_param_indices
                   if i < len(args) and isinstance(args[i], torch.Tensor)}
        with torch.autograd.graph.saved_tensors_hooks(
                lambda t: self.conductor.pack_activation(layer_index, t) if _view_key(t) in targets else t,
                self.conductor.unpack_activation):
            return run()

    def forward(self, *args, **kwargs):
        args = _kwargs_to_args(self.orig_forward if self.checkpoint is None else self.checkpoint.forward, args, kwargs)

        if not torch.is_grad_enabled():
            # inference / frozen: no backward flows, so no boundaries are needed. Schedule, run
            # and record inline.
            if self.layer_index == 0:
                self.conductor.start_forward(backward_follows=False)
            self.conductor.before_layer(self.layer_index, is_forward=True)
            # still through run_block: under no_grad the checkpoint inside it is a passthrough, and this
            # keeps sampling on the compiled path (dynamo traces a separate inference variant)
            output = self.run_block(*args)
            self.conductor.after_layer(self.layer_index, list(args))
            return output

        if self.layer_index == 0:
            self.conductor.start_forward(backward_follows=True)

        args = _apply_boundary(LoadBoundary, self.conductor, self.layer_index, args)
        output = self.__run_block(args)
        output_tuple = output if isinstance(output, tuple) else (output,)
        output_tuple = _apply_boundary(EvictBoundary, self.conductor, self.layer_index, output_tuple)
        return output_tuple if isinstance(output, tuple) else output_tuple[0]


def create_checkpoint(
        orig_module: nn.Module,
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
        conductor.add_layer(orig_module)

    if conductor is not None and conductor.offload_activated():
        # Compiled, the boundary layer compiles the block together with its checkpoint, so orig_module must
        # not be compiled separately here - that would put the checkpoint back outside the compiled region
        # and force a full recompute.
        if compile:
            return BoundaryOffloadCheckpointLayer(
                orig_module=orig_module,
                orig_forward=None,
                conductor=conductor,
                layer_index=layer_index,
                checkpointing=checkpointing,
                included_offload_param_indices=included_offload_param_indices,
                compile=True,
            )
        else:
            #only patch forward() if possible. Inserting layers is necessary for torch.compile, but causes issues with at least 1 text encoder model. we don't compile text encoders
            layer = BoundaryOffloadCheckpointLayer(
                orig_module=None,
                orig_forward=orig_module.forward,
                conductor=conductor,
                layer_index=layer_index,
                checkpointing=checkpointing,
                included_offload_param_indices=included_offload_param_indices,
                compile=False,
            )
            orig_module.forward = layer.forward
            return orig_module
    else:
        if compile:
            layer = CheckpointLayer(orig_module=orig_module, orig_forward=None, checkpointing=checkpointing)
            #do compile the checkpointing layer - slightly faster
            layer.compile(fullgraph=True)
            return layer
        else:
            layer = CheckpointLayer(orig_module=None, orig_forward=orig_module.forward, checkpointing=checkpointing)
            orig_module.forward = layer.forward
            return orig_module

def _create_checkpoints_for_module_list(
        module_list: nn.ModuleList,
        include_from_offload_param_names: list[str],
        conductor: LayerOffloadConductor,
        checkpointing: bool,
        layer_index: int,
        compile: bool,
) -> int:

    for i, layer in enumerate(module_list):
        if isinstance(module_list[i], BaseCheckpointLayer):
            continue
        module_list[i] = create_checkpoint(
                layer,
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

    # Offloading requires checkpointing. A block's backward reads its weights through SavedVariables taken
    # during the forward, which hold a shallow copy - repointing param.data at a reloaded buffer never
    # redirects them - while the conductor recycles weight buffers between layers, so by then that buffer
    # can already hold a different layer's weights: plausible but wrong gradients, not an error. The
    # recompute re-reads the weights after the layer has been loaded back in. Compiled blocks escape this
    # today (AOTAutograd passes parameters as graph inputs, read at call time), but that is a calling
    # convention and not a guarantee. Frozen parts run no backward through the block and are exempt.
    if offload and not checkpointing and part.train:
        raise NotImplementedError("offloading requires gradient checkpointing")

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


def enable_checkpointing_for_ideogram_transformer(
        model: nn.Module,
        config: TrainConfig,
        part: TrainModelPartConfig,
) -> LayerOffloadConductor | None:
    return enable_checkpointing(model, config, part, config.compile, [
        (model.layers, ["hidden_states"]),
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
