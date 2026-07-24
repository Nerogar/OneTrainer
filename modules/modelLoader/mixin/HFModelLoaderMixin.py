import json
import logging
import os
import queue
import threading
from abc import ABCMeta
from itertools import repeat

from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.DataType import DataType
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import (
    is_quantized_module,
    is_quantized_parameter,
    replace_linear_with_quantized_layers,
)
from modules.util.torch_util import mem_pool_context

import torch
from torch import nn

from diffusers import GGUFQuantizationConfig
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import rename_source_key

import accelerate
import huggingface_hub
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

# huggingface_hub 1.16+ uses httpx, which logs every HTTP request/response at INFO level.
logging.getLogger("httpx").setLevel(logging.WARNING)

# reader threads striping the checkpoint into host RAM while the main thread does H2D + inline quant
STREAM_READER_THREADS = 4


def __stream_reader(
        tid: int,
        nthreads: int,
        work: list[tuple],
        key_to_file: dict[str, str],
        source_key_map: dict[str, str] | None,
        out_queue: queue.Queue,
        done,
):
    # prefetch reader thread: reads a stripe of the work list into host RAM and feeds the bounded queue. Each thread
    # owns its safe_open handles (a handle is not safe for concurrent get_tensor).
    thread_handles: dict[str, object] = {}
    try:
        for i in range(tid, len(work), nthreads):
            item = work[i]
            path = key_to_file[item[0]]
            handle = thread_handles.get(path)
            if handle is None:
                handle = thread_handles[path] = safe_open(path, framework="pt", device="cpu")
            # cache key is the renamed module-layout key; the file stores the original, so read by the original
            # (identity when no rename map was built).
            read_key = source_key_map.get(item[0], item[0]) if source_key_map else item[0]
            # get_tensor returns a lazy mmap view; .clone() forces the read off disk into host RAM
            out_queue.put((item, handle.get_tensor(read_key).clone()))
    except Exception as e:
        out_queue.put(e)
    finally:
        out_queue.put(done)


def _intended_float_dtype(
        module: nn.Module,
        module_name: str,
        tensor_name: str,
        dtype: DataType,
        train_dtype: DataType,
        keep_in_fp32_modules: list[str],
) -> torch.dtype | None:
    # target dtype for a streamed float tensor, or None to leave it unchanged. A param the quantizer will pack keeps
    # its dtype (the quantizer converts it); keep-in-fp32 modules and a quantized component's leftover params go to
    # train_dtype; everything else to the weight dtype.
    if is_quantized_parameter(module, tensor_name):
        return None
    if dtype.is_quantized() or module_name in keep_in_fp32_modules:
        # a caller without a train_dtype yet (budget sizing) gets None -> the budget over-estimates these from the
        # fp32 skeleton; the stream-time caller always passes a real train_dtype.
        return train_dtype.torch_dtype() if train_dtype is not None else None
    return dtype.torch_dtype()


def _stamp_skeleton_float_dtypes(
        sub_module: nn.Module,
        dtype: DataType,
        train_dtype: DataType,
        keep_in_fp32_modules: list[str],
):
    # stamp each meta-skeleton float param with the dtype the stream will give it, so the offload VRAM budget (which
    # sizes the still-meta skeleton) measures the real post-load footprint, not init_empty_weights' fp32 default. Free:
    # a meta tensor holds no data, so .to() only rewrites its declared dtype. Uses the same _intended_float_dtype helper
    # as the stream-time cast so the two agree; quantized weights are left alone (sized via predict_offload_bytes).
    # Buffers are not stamped: they never enter the offload budget.
    for name, module in sub_module.named_modules():
        module_name = name.split(".")[-1]
        for tensor_name, param in module.named_parameters(recurse=False):
            if not torch.is_floating_point(param):
                continue
            target = _intended_float_dtype(module, module_name, tensor_name, dtype, train_dtype, keep_in_fp32_modules)
            if target is not None and param.dtype != target:
                param.data = param.data.to(dtype=target)


def stream_module_from_checkpoint(
        module: nn.Module,
        device: torch.device,
        key_to_file: dict[str, str],
        dtype: DataType,
        train_dtype: DataType,
        keep_in_fp32_modules: list[str],
        tied_weights_keys: dict[str, str] | None,
        quantize: bool,
        key_prefix: str = "",
        source_key_map: dict[str, str] | None = None,
        part_name: str | None = None,
        dest_pool=None,
):
    # Fill a meta skeleton by streaming its checkpoint weights one tensor at a time, so the full checkpoint never lands
    # in RAM. key_prefix scopes the lookup to one sub-module; keys stay checkpoint-absolute. dest_pool routes
    # non-quantized weights straight into a MemPool (quantized modules pack in the default pool).
    def dest_pool_for(sub_module):
        return dest_pool if (dest_pool is not None and not is_quantized_module(sub_module)) else None

    # flat work list of every checkpoint-backed skeleton tensor, so the reader threads below can drive the reads.
    work = []  # (key, sub_module, tensor_name, is_buffer, module_name)
    for name, sub_module in module.named_modules():
        module_name = name.split(".")[-1]
        # gradient checkpointing in compile mode wraps each block in a CheckpointLayer, inserting a ".checkpoint."
        # level into the live path; the checkpoint keys have none, so strip it before lookup (as LoRAModule does).
        lookup_name = name.replace(".checkpoint.", ".")
        for tensor_name, param in list(sub_module.named_parameters(recurse=False)):
            key = ".".join(p for p in (key_prefix, lookup_name, tensor_name) if p)
            if key in key_to_file and param.is_meta:
                work.append((key, sub_module, tensor_name, False, module_name))
        for tensor_name, _buffer in list(sub_module.named_buffers(recurse=False)):
            # non-persistent buffers (rotary inv_freq etc.) are config-derived, not stored in the checkpoint
            if tensor_name in sub_module._non_persistent_buffers_set:
                continue
            # no is_meta guard (unlike params): init_empty_weights materializes persistent buffers as REAL init values,
            # so is_meta can't mean "not yet filled" -- always stream, else the init value survives (mis-normalizing the VAE).
            key = ".".join(p for p in (key_prefix, lookup_name, tensor_name) if p)
            if key in key_to_file:
                work.append((key, sub_module, tensor_name, True, module_name))

    # place() lands one tensor: cast floats to their intended dtype (quantizer-packed params keep theirs), move to the
    # compute device, quantize inline once a layer's weight arrives so VRAM never holds the whole unquantized module.
    # bar: one tick per streamed tensor; only a whole-module stream (part_name set) shows it, per-layer conductor calls stay silent.
    bar = tqdm(total=len(work), unit="tensor", desc=f"streaming {part_name}", leave=False, smoothing=0.05) \
        if part_name is not None else None

    def quantize_if_ready(sub_module):
        # quantize a module whose weight has landed (no longer meta): quantize() self-guards against a second call, so
        # firing it the moment the weight arrives (rather than in the batch pass quantize_layers() does) is always safe.
        if isinstance(sub_module, QuantizedModuleMixin) and not sub_module.weight.is_meta:
            sub_module.compute_dtype = train_dtype.torch_dtype()
            sub_module.quantize(device=device)

    def place(item, value):
        _key, sub_module, tensor_name, is_buffer, module_name = item
        # tensors that will be quantized stay at their original dtype (the quantizer converts them); everything else is
        # cast to its intended dtype here.
        if torch.is_floating_point(value):
            target = _intended_float_dtype(sub_module, module_name, tensor_name, dtype, train_dtype, keep_in_fp32_modules)
            if target is not None:
                value = value.to(dtype=target)
        with mem_pool_context(dest_pool_for(sub_module)):
            set_module_tensor_to_device(sub_module, tensor_name, device, value=value, dtype=value.dtype)
        # quantize outside the pool context so a quantized module's dequant scratch stays in the default pool
        if quantize:
            quantize_if_ready(sub_module)
        if bar is not None:
            bar.update(1)

    # reader threads stripe the work list into host RAM and feed a bounded queue; the main thread drains it and does
    # H2D + inline quantize on the default stream. Both the parallel reads and overlapping them with the GPU work are
    # wins. Each reader clones the tensor off its mmap and drops its safetensors handles when it exits (right after its
    # stripe), so the file mmaps are released early rather than pinned until first use -- keeps page-cache pressure
    # down. Tensors may land out of order -- place() addresses each by name and inline quant is order-free.
    nthreads = STREAM_READER_THREADS
    out_queue: queue.Queue = queue.Queue(maxsize=2 * nthreads)
    done = object()

    threads = [
        threading.Thread(
            target=__stream_reader,
            args=(tid, nthreads, work, key_to_file, source_key_map, out_queue, done),
            name=f"stream-reader-{tid}", daemon=True,
        )
        for tid in range(nthreads)
    ]
    for t in threads:
        t.start()
    finished = 0
    while finished < nthreads:
        got = out_queue.get()
        if got is done:
            finished += 1
        elif isinstance(got, Exception):
            raise got
        else:
            place(*got)
    for t in threads:
        t.join()

    # tied weights (e.g. Qwen3 lm_head <-> embed_tokens) are saved once, so the target stays meta; fill it with an
    # independent clone of the source (not an alias -- in-place quantize would corrupt both), then quantize. Both keys
    # are module-root-relative, so whole-module streams only (key_prefix == "").
    if not key_prefix:
        for target_key, source_key in (tied_weights_keys or {}).items():
            parent_path, _, target_name = target_key.rpartition(".")
            target_module = module.get_submodule(parent_path)
            if target_module._parameters[target_name].is_meta:
                source = module.get_parameter(source_key)
                with mem_pool_context(dest_pool_for(target_module)):
                    set_module_tensor_to_device(
                        target_module, target_name, device, value=source.detach().clone(), dtype=source.dtype)
                if quantize:
                    quantize_if_ready(target_module)

    # non-persistent buffers (rotary inv_freq etc.) are skipped above but materialized REAL on cpu by init_empty_weights;
    # move them to the device so the forward doesn't see cpu buffers vs device activations. Whole-module streams only.
    if not key_prefix and device.type != "meta":
        for sub_module in module.modules():
            for buffer_name in sub_module._non_persistent_buffers_set:
                buffer = sub_module._buffers.get(buffer_name)
                if buffer is not None and not buffer.is_meta:
                    with mem_pool_context(dest_pool_for(sub_module)):
                        sub_module._buffers[buffer_name] = buffer.to(device)

    if bar is not None:
        bar.close()


class HFModelLoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    # ===== LEGACY (non-streaming) load path -- used only when Stream From Disk is off =====
    def __load_sub_module_legacy(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            keep_in_fp32_modules: list[str] | None,
            quantization: QuantizationConfig | None,
            pretrained_model_name_or_path: str,
            subfolder: str | None,
            model_filename: str,
            pytorch_model_filename: str | None,
            shard_index_filename: str,
    ):
        if keep_in_fp32_modules is None:
            keep_in_fp32_modules = []

        replace_linear_with_quantized_layers(sub_module, dtype, keep_in_fp32_modules, quantization, copy_parameters=False)

        is_local = os.path.isdir(pretrained_model_name_or_path)

        if is_local:
            if subfolder:
                full_shard_index_filename = os.path.join(pretrained_model_name_or_path, subfolder, shard_index_filename)
            else:
                full_shard_index_filename = os.path.join(pretrained_model_name_or_path, shard_index_filename)
            if not os.path.isfile(full_shard_index_filename):
                full_shard_index_filename = None
        else:
            try:
                full_shard_index_filename = huggingface_hub.hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=shard_index_filename,
                )
            except EntryNotFoundError:
                full_shard_index_filename = None

        is_sharded = full_shard_index_filename is not None

        if is_sharded:
            with open(full_shard_index_filename, "r") as f:
                index_file = json.loads(f.read())
                safetensors_filenames = sorted(set(index_file["weight_map"].values()))
        else:
            safetensors_filenames = [model_filename]

        state_dict = {}
        is_torch_pickle = False

        if is_local:
            if subfolder:
                full_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in
                                  safetensors_filenames]
            else:
                full_filenames = [os.path.join(pretrained_model_name_or_path, f) for f in safetensors_filenames]

            if any(not os.path.isfile(f) for f in full_filenames):
                # fall back to the pytorch_model_filename
                full_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, pytorch_model_filename)]
                is_torch_pickle = True
        else:
            try:
                full_filenames = [huggingface_hub.hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=f,
                ) for f in safetensors_filenames]
            except EntryNotFoundError as _:
                # fall back to the pytorch_model_filename
                full_filenames = [huggingface_hub.hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=pytorch_model_filename,
                )]
                is_torch_pickle = True

        if is_torch_pickle:
            for f in full_filenames:
                file_state_dict = torch.load(f, weights_only=True)
                while 'state_dict' in file_state_dict:
                    file_state_dict = file_state_dict['state_dict']
                state_dict |= file_state_dict
        else:
            for f in full_filenames:
                state_dict |= load_file(f)

        if hasattr(sub_module, '_fix_state_dict_keys_on_load'):
            sub_module._fix_state_dict_keys_on_load(state_dict)

        #some checkpoints (e.g. Ernie's Mistral3 text encoder, Qwen's Qwen2_5_VL text encoder) were saved with an
        #older module layout than the one transformers builds from the config in this version. transformers' own
        #from_pretrained applies the same renaming via its checkpoint conversion registry, so we reuse it here.
        #diffusers sub-modules have no such registry (their config is a plain FrozenDict, no model_type), and
        #never need this renaming.
        weight_renamings = get_checkpoint_conversion_mapping(sub_module.config.model_type) \
            if hasattr(sub_module.config, 'model_type') else None
        if weight_renamings:
            meta_state_dict = sub_module.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k, _ = rename_source_key(
                    k, weight_renamings, [], prefix=sub_module.base_model_prefix, meta_state_dict=meta_state_dict,
                )
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        #tensors that will be quantized are loaded at their original dtype. non-quantized tensors are converted
        #to their intended dtype here
        #TODO the following code requires quite a few workarounds by now. Is there a better way?
        for key, value in state_dict.items():
            module = sub_module
            tensor_name = key
            module_name = None
            key_splits = tensor_name.split(".")
            for split in key_splits[:-1]:
                module = getattr(module, split)
                module_name = split
            tensor_name = key_splits[-1]

            is_buffer = tensor_name in module._buffers
            if not is_buffer and tensor_name not in module._parameters:
                continue
            old_value = module._buffers[tensor_name] if is_buffer else module._parameters[tensor_name]

            if torch.is_floating_point(old_value):
                old_type = type(old_value)
                if not is_quantized_parameter(module, tensor_name):
                    if dtype.is_quantized() or module_name in keep_in_fp32_modules:
                        value = value.to(dtype=train_dtype.torch_dtype())
                    else:
                        value = value.to(dtype=dtype.torch_dtype())

                new_value = old_type(value)

                if is_buffer:
                    module._buffers[tensor_name].data = new_value
                else:
                    module._parameters[tensor_name] = new_value

        del state_dict

        #tied weights (e.g. T5EncoderModel's encoder.embed_tokens.weight <-> shared.weight, or
        #Qwen3ForCausalLM's lm_head.weight <-> model.embed_tokens.weight) are saved only once in the checkpoint,
        #so the tied key above is never assigned and stays an empty meta tensor. populate it by cloning the
        #source weight that was actually loaded, rather than aliasing the same Parameter object: aliasing would
        #make a later in-place quantization of one side (e.g. quantize_layers() quantizing lm_head) silently
        #corrupt the other (e.g. the embedding table), since both attribute paths would refer to the same object.
        tied_weights_keys = getattr(sub_module, '_tied_weights_keys', None)
        if tied_weights_keys is not None:
            for target_key, source_key in tied_weights_keys.items():
                module = sub_module
                *parents, tensor_name = target_key.split(".")
                for p in parents:
                    module = getattr(module, p)
                if module._parameters[tensor_name].is_meta:
                    source = sub_module.get_parameter(source_key)
                    module._parameters[tensor_name] = type(module._parameters[tensor_name])(source)

        return sub_module
    # ===== end LEGACY load path =====

    def _load_transformers_sub_module(
            self,
            module_type,
            dtype: DataType,
            train_dtype: DataType,
            pretrained_model_name_or_path: str,
            subfolder: str = "",
            stream_from_disk: bool = False,
    ):
        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
        }
        config, model_kwargs = module_type.config_class.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            user_agent=user_agent,
        )

        with accelerate.init_empty_weights():
            sub_module = module_type(config)

        if not stream_from_disk:
            # LEGACY fallback: whole-checkpoint-into-RAM load
            return self.__load_sub_module_legacy(
                sub_module=sub_module,
                dtype=dtype,
                train_dtype=train_dtype,
                keep_in_fp32_modules=module_type._keep_in_fp32_modules,
                quantization=None,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                subfolder=subfolder,
                model_filename="model.safetensors",
                pytorch_model_filename="pytorch_model.bin",
                shard_index_filename="model.safetensors.index.json",
            )

        keep_in_fp32_modules = module_type._keep_in_fp32_modules or []
        replace_linear_with_quantized_layers(sub_module, dtype, keep_in_fp32_modules, None, copy_parameters=False)
        # stamp with train_dtype=None: the per-part train_dtype is only known at materialize, so keep-in-fp32 and
        # quantized-leftover params stay fp32 in the skeleton (a safe budget over-estimate); see _intended_float_dtype
        _stamp_skeleton_float_dtypes(sub_module, dtype, None, keep_in_fp32_modules)

        key_to_file = self.__resolve_shard_key_to_file(
            pretrained_model_name_or_path, subfolder,
            model_filename="model.safetensors",
            shard_index_filename="model.safetensors.index.json",
        )

        # some checkpoints (e.g. Ernie's Mistral3, Qwen's Qwen2_5_VL text encoders) were saved with an older module
        # layout than transformers builds from the config now. Reuse transformers' own checkpoint conversion registry
        # to rename the checkpoint keys to the module's layout so the streamed lookup finds them. diffusers sub-modules
        # have no such registry (plain FrozenDict config, no model_type) and never need this.
        weight_renamings = get_checkpoint_conversion_mapping(sub_module.config.model_type) \
            if hasattr(sub_module.config, 'model_type') else None
        source_key_map = None
        if weight_renamings:
            meta_state_dict = sub_module.state_dict()
            renamed_key_to_file = {}
            # the rename maps each checkpoint key to the module's layout so the streamed lookup and the offload cache
            # find it; the file itself still stores the original key, so keep renamed->original to read the tensor.
            source_key_map = {}
            for key, file in key_to_file.items():
                renamed = rename_source_key(
                    key, weight_renamings, [], prefix=sub_module.base_model_prefix, meta_state_dict=meta_state_dict,
                )[0]
                renamed_key_to_file[renamed] = file
                source_key_map[renamed] = key
            key_to_file = renamed_key_to_file

        return self.__finish_sub_module_load(
            sub_module, dtype, train_dtype, keep_in_fp32_modules, key_to_file, source_key_map=source_key_map)

    def __resolve_shard_key_to_file(
            self,
            pretrained_model_name_or_path: str,
            subfolder: str,
            model_filename: str,
            shard_index_filename: str,
    ) -> dict[str, str]:
        # map every checkpoint tensor key to the local safetensors file that holds it (downloading shards from the
        # hub if the source is a repo id), so the streaming fill can read each tensor on demand.
        is_local = os.path.isdir(pretrained_model_name_or_path)

        def resolve(filename: str) -> str | None:
            # return a local path to `filename` (downloading it from the hub if needed), or None if it is absent
            if is_local:
                if subfolder:
                    path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
                else:
                    path = os.path.join(pretrained_model_name_or_path, filename)
                return path if os.path.isfile(path) else None
            try:
                return huggingface_hub.hf_hub_download(
                    repo_id=pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
            except EntryNotFoundError:
                return None

        key_to_file = {}

        index_path = resolve(shard_index_filename)
        if index_path is not None:
            with open(index_path, "r") as f:
                weight_map = json.loads(f.read())["weight_map"]
            shard_paths = {shard: resolve(shard) for shard in set(weight_map.values())}
            for key, shard in weight_map.items():
                key_to_file[key] = shard_paths[shard]
            return key_to_file

        # non-sharded: prefer the full-precision safetensors, fall back to the fp16 variant (some older repos, e.g.
        # stable-diffusion-inpainting, ship only *.fp16.safetensors next to legacy pickle .bin files). Pickle .bin
        # weights are not supported -- safe_open needs safetensors for random per-tensor reads.
        fp16_filename = model_filename.replace(".safetensors", ".fp16.safetensors")
        full_filename = resolve(model_filename) or resolve(fp16_filename)
        if full_filename is None:
            location = f"{pretrained_model_name_or_path}/{subfolder}" if subfolder else pretrained_model_name_or_path
            raise FileNotFoundError(
                f"No safetensors weights found for '{location}' (looked for {model_filename} and {fp16_filename}). "
                f"Only pickle .bin checkpoints are present, which are not supported; convert the model to "
                f"safetensors.")
        with safe_open(full_filename, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 -- safe_open handle, not a dict
                key_to_file[key] = full_filename

        return key_to_file

    def _load_diffusers_sub_module(
            self,
            module_type,
            dtype: DataType,
            train_dtype: DataType,
            pretrained_model_name_or_path: str,
            subfolder: str | None = None,
            quantization: QuantizationConfig | None = None,
            stream_from_disk: bool = False,
    ):
        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
        }
        config, unused_kwargs, commit_hash = module_type.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            user_agent=user_agent,
        )

        with accelerate.init_empty_weights():
            sub_module = module_type.from_config(config)

        if not stream_from_disk:
            # LEGACY fallback: whole-checkpoint-into-RAM load
            return self.__load_sub_module_legacy(
                sub_module=sub_module,
                dtype=dtype,
                train_dtype=train_dtype,
                keep_in_fp32_modules=module_type._keep_in_fp32_modules,
                quantization=quantization,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                subfolder=subfolder,
                model_filename="diffusion_pytorch_model.safetensors",
                pytorch_model_filename="diffusion_pytorch_model.bin",
                shard_index_filename="diffusion_pytorch_model.safetensors.index.json",
            )

        keep_in_fp32_modules = module_type._keep_in_fp32_modules or []
        replace_linear_with_quantized_layers(sub_module, dtype, keep_in_fp32_modules, quantization, copy_parameters=False)
        # stamp with train_dtype=None: the per-part train_dtype is only known at materialize, so keep-in-fp32 and
        # quantized-leftover params stay fp32 in the skeleton (a safe budget over-estimate); see _intended_float_dtype
        _stamp_skeleton_float_dtypes(sub_module, dtype, None, keep_in_fp32_modules)

        key_to_file = self.__resolve_shard_key_to_file(
            pretrained_model_name_or_path, subfolder,
            model_filename="diffusion_pytorch_model.safetensors",
            shard_index_filename="diffusion_pytorch_model.safetensors.index.json",
        )

        # diffusers renamed deprecated attention-block weights (query->to_q etc.); older single-file checkpoints still
        # use the old names. _fix_state_dict_keys_on_load rewrites them to the current layout, and since it only
        # renames dict keys, applying it to the key->file map matches applying it to a state_dict. No-op for modern
        # architectures.
        if hasattr(sub_module, '_fix_state_dict_keys_on_load'):
            sub_module._fix_state_dict_keys_on_load(key_to_file)

        return self.__finish_sub_module_load(
            sub_module, dtype, train_dtype, keep_in_fp32_modules, key_to_file)

    def __finish_sub_module_load(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            keep_in_fp32_modules: list[str],
            key_to_file: dict[str, str],
            source_key_map: dict[str, str] | None = None,
    ):
        tied_weights_keys = getattr(sub_module, "_tied_weights_keys", None)

        # module/key_prefix let the layer-offload conductor reuse this same closure to stream one layer at a time
        # (module=that layer, key_prefix=its path in the checkpoint) as well as the non-layer remainder
        # (module=the whole sub-module, key_prefix=""). Whole-module callers pass neither and stream everything.
        def materialize_fn(
                module: nn.Module, device: torch.device, train_dtype: DataType, key_prefix: str = "",
                part_name: str | None = None, dest_pool=None):
            stream_module_from_checkpoint(
                module, device, key_to_file, dtype, train_dtype,
                keep_in_fp32_modules, tied_weights_keys, quantize=True, key_prefix=key_prefix,
                source_key_map=source_key_map, part_name=part_name, dest_pool=dest_pool)

        return sub_module, materialize_fn

    def __convert_sub_module_to_dtype(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            keep_in_fp32_modules: list[str] | None,
            quantization: QuantizationConfig | None,
    ):
        if keep_in_fp32_modules is None:
            keep_in_fp32_modules = []

        replace_linear_with_quantized_layers(sub_module, dtype, keep_in_fp32_modules, quantization, copy_parameters=True)

        for module_name, module in sub_module.named_modules():
            param_iter = [(x, y[0], y[1]) for x, y in zip(repeat(False), module._parameters.items(), strict=False)]
            buffer_iter = [(x, y[0], y[1]) for x, y in zip(repeat(True), module._buffers.items(), strict=False)]
            for is_buffer, tensor_name, value in param_iter + buffer_iter:
                if value is not None and torch.is_floating_point(value):
                    old_type = type(value)
                    if not is_quantized_parameter(module, tensor_name):
                        if dtype.is_quantized() or module_name in keep_in_fp32_modules:
                            value = value.to(dtype=train_dtype.torch_dtype())
                        else:
                            value = value.to(dtype=dtype.torch_dtype())

                        value = old_type(value)

                        if is_buffer:
                            module._buffers[tensor_name].data = value
                        else:
                            module._parameters[tensor_name] = value

        return sub_module

    def _convert_transformers_sub_module_to_dtype(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            quantization: QuantizationConfig | None = None,
    ):
        module_type = type(sub_module)

        return self.__convert_sub_module_to_dtype(
            sub_module,
            dtype,
            train_dtype,
            module_type._keep_in_fp32_modules,
            quantization,
        )

    def _convert_diffusers_sub_module_to_dtype(
            self,
            sub_module: nn.Module,
            dtype: DataType,
            train_dtype: DataType,
            quantization: QuantizationConfig | None = None,
    ):
        return self.__convert_sub_module_to_dtype(
            sub_module,
            dtype,
            train_dtype,
            None,
            quantization,
        )

    def _load_transformer(
            self,
            module_type,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            quantization: QuantizationConfig,
            config: str | None = None,
            stream_from_disk: bool = False,
    ):
        # a single-file (optionally GGUF-quantized) checkpoint is loaded directly, using
        # a separate repo to source the model config if the checkpoint doesn't carry one;
        # otherwise the transformer is loaded from its subfolder in the base model repo.
        # Always returns a (transformer, materialize_fn) pair -- materialize_fn None when not streamed -- so callers
        # pass stream_from_disk through.
        if transformer_model_name:
            single_file_kwargs = {}
            if config is not None:
                single_file_kwargs["config"] = config
                single_file_kwargs["subfolder"] = "transformer"

            transformer = module_type.from_single_file(
                transformer_model_name,
                **single_file_kwargs,
                #avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
            return transformer, None
        elif stream_from_disk:
            # stream from disk: meta skeleton + materialize closure; weights are streamed and quantized to the compute
            # device on use and evicted back to meta afterwards, so the full unquantized module never lands in RAM.
            # train_dtype is applied per-materialize, not here.
            return self._load_diffusers_sub_module(
                module_type,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
                stream_from_disk=True,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                module_type,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )
            return transformer, None

    def _load_text_encoder(
            self,
            module_type,
            dtype: DataType,
            train_dtype: DataType,
            base_model_name: str,
            subfolder: str,
            stream_from_disk: bool = False,
    ):
        # text encoders have no single-file override and always load from their subfolder. Always returns a
        # (text_encoder, materialize_fn) pair -- materialize_fn None when not streamed -- mirroring _load_transformer.
        # dtype/train_dtype are explicit rather than a weight_dtypes bundle since a model can hold several encoders
        # (text_encoder, text_encoder_2, ...) with differing dtypes.
        if stream_from_disk:
            return self._load_transformers_sub_module(
                module_type,
                dtype,
                train_dtype,
                base_model_name,
                subfolder,
                stream_from_disk=True,
            )
        else:
            text_encoder = self._load_transformers_sub_module(
                module_type,
                dtype,
                train_dtype,
                base_model_name,
                subfolder,
            )
            return text_encoder, None

    def _load_vae(
            self,
            module_type,
            dtype: DataType,
            train_dtype: DataType,
            base_model_name: str,
            vae_model_name: str,
    ):
        # a separate vae repo overrides the base model's vae subfolder when given. train_dtype is explicit
        # since some models (e.g. SDXL) upgrade the vae to fallback_train_dtype to avoid fp16 overflow
        if vae_model_name:
            return self._load_diffusers_sub_module(
                module_type,
                dtype,
                train_dtype,
                vae_model_name,
            )
        else:
            return self._load_diffusers_sub_module(
                module_type,
                dtype,
                train_dtype,
                base_model_name,
                "vae",
            )
