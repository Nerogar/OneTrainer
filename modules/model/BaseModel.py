from abc import ABCMeta
from collections.abc import Callable
from contextlib import nullcontext
from uuid import uuid4

from modules.module.EMAModule import EMAModuleWrapper
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert_util import qkv_fusion
from modules.util.disk_stream import stream_module_to
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor
from modules.util.modelSpec.ModelSpec import ModelSpec
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.torch_util import create_mem_pool, device_equals, mem_pool_context, supports_mem_pool, torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
from torch.optim import Optimizer

from transformers import PreTrainedTokenizer


class BaseModelEmbedding:
    def __init__(
            self,
            uuid: str,
            placeholder: str,
            vector: torch.Tensor | None,
            is_output_embedding: bool,
    ):
        self.uuid = uuid
        self.token_count = vector.shape[0] if vector is not None else 0
        self.placeholder = placeholder
        self.is_output_embedding = is_output_embedding

        if vector is not None:
            if is_output_embedding:
                self.vector = torch.zeros_like(vector).requires_grad_(False)
                self.output_vector = vector.to(dtype=torch.float32)
                self.original_output_vector_std = self.output_vector.std(dim=1).mean()
            else:
                self.vector = vector
                self.output_vector = None
                self.original_output_vector_std = None
        else:
            self.vector = None
            self.output_vector = None
            self.original_output_vector_std = None

        self.text_tokens = [f"<{uuid4()}>" for _ in range(self.token_count)]
        self.joint_text_tokens = ''.join(self.text_tokens)
        self.joint_tokens_cache = None  # a cache for the joint_text_tokens when sent through the tokenizer

    def requires_grad(self) -> bool:
        if self.is_output_embedding:
            return self.output_vector.requires_grad
        else:
            return self.vector.requires_grad

    def requires_grad_(self, requires_grad: bool):
        if self.is_output_embedding:
            self.output_vector.requires_grad_(requires_grad)
        else:
            self.vector.requires_grad_(requires_grad)


class BaseModel(metaclass=ABCMeta):
    model_type: ModelType
    parameters: NamedParameterGroupCollection | None
    optimizer: Optimizer | None
    optimizer_state_dict: dict | None
    param_group_mapping: list[str] | None
    ema: EMAModuleWrapper
    ema_state_dict: dict | None
    train_progress: TrainProgress
    model_spec: ModelSpec | None
    train_config: TrainConfig | None
    embedding_state_dicts: dict[str, dict[str, Tensor]] | None
    autocast_context: torch.autocast | nullcontext
    train_dtype: DataType
    cache_in_ram: dict[str, bool]
    offload_conductor: dict[str, LayerOffloadConductor]
    materialize_fn: dict[str, Callable]

    def __init__(
            self,
            model_type: ModelType,
    ):
        self.model_type = model_type
        self.parameters = None
        self.cache_in_ram = {}
        self.offload_conductor = {}
        self.materialize_fn = {}
        self.optimizer = None
        self.optimizer_state_dict = None
        self.param_group_mapping = None
        self.ema_state_dict = None
        self.train_progress = TrainProgress()
        self.model_spec = None
        self.train_config = None
        self.embedding_state_dicts = {}
        self.autocast_context = nullcontext()
        self.train_dtype = DataType.FLOAT_32

        self._mem_pools = {}

    @property
    def train_device(self) -> torch.device:
        return torch.device(self.train_config.train_device)

    @property
    def temp_device(self) -> torch.device:
        return torch.device(self.train_config.temp_device)

    def materialize(self, *parts: str):
        # Move `parts` onto train_device.
        for part in parts:
            self._move_part(part, self.train_device)

    def evict(self, *parts: str):
        # Move `parts` onto temp_device. No parts given -> every component in ModelType.model_parts().
        for part in parts or self.model_type.model_parts():
            self._move_part(part, self.temp_device)
        torch_gc()

    def materialize_only(self, *parts: str):
        # Materialize exactly `parts` on train_device; evict every other component in ModelType.model_parts()
        # to temp_device. Lets a caller state what it needs now without tracking what to evict first.
        # Evicts before materializing, so the two sets are never resident on train_device at once.
        # Skipped (rather than passed as evict()) when empty, since evict() with no parts means "evict all".
        to_evict = [part for part in self.model_type.model_parts() if part not in parts]
        if to_evict:
            self.evict(*to_evict)
        self.materialize(*parts)

    def materialize_only_text_encoders(self):
        # Materialize all of this model's text encoders on train_device, evicting everything else. Samplers
        # call this before encode_text, which reads every text encoder the model has.
        self.materialize_only(*self.model_type.text_encoder_parts())

    def _move_part(self, part: str, device: torch.device):
        # Move a component (`part`, or `part_1` for the first of several split text encoders) and its LoRA. The
        # dispatch below routes through an offload conductor and/or a disk-stream materialize closure if present.
        stem = f"{part}_1" if hasattr(self, f"{part}_1") else part

        conductor = self.offload_conductor.get(stem)
        materialize_fn = self.materialize_fn.get(stem)
        cache_in_ram = self.cache_in_ram.get(stem, True)

        if conductor is not None:
            if device_equals(device, self.train_device):
                train_dtype = getattr(self, f"{stem}_train_dtype", self.train_dtype)
                conductor.materialize(
                    train_dtype, name=part, materialize_fn=materialize_fn,
                    cache_in_ram=cache_in_ram)
            else:
                to_meta = materialize_fn is not None and not cache_in_ram
                conductor.evict(to_meta=to_meta)
        elif materialize_fn is not None:
            streamed_component = getattr(self, stem)
            train_dtype = getattr(self, f"{stem}_train_dtype", self.train_dtype)
            stream_module_to(
                streamed_component, device, materialize_fn, train_dtype,
                cache_in_ram=cache_in_ram, name=part, temp_device=self.temp_device)

        # move into the shared stem pool: the base component itself (unless a conductor or stream owns its move) plus
        # the LoRA. getattr(self, stem) is None for a part in model_parts() that was never populated (e.g. an omitted
        # text encoder), so it drops out below.
        to_move = []
        if conductor is None and materialize_fn is None:
            to_move.append(getattr(self, stem))
        lora = getattr(self, f"{stem}_lora", None)
        to_move.append(lora)
        to_move = [module for module in to_move if module is not None]
        if not to_move:
            return

        if supports_mem_pool(device):
            # The component (when not conductor/stream-managed) and its LoRA share a per-stem MemPool so both release
            # together on evict, keeping the LoRA's small tensors from pinning freed default-pool segments across the
            # part's evict/reload cycle. A conductor keeps its own pool, so the stem pool then holds only the LoRA.
            pool = self._mem_pools.get(stem)
            if pool is None:
                pool = self._mem_pools[stem] = create_mem_pool(device)
            with mem_pool_context(pool):
                for module in to_move:
                    module.to(device=device)
        else:
            # the target has no MemPool (CPU): move normally and drop this stem's pool from the earlier GPU move,
            # so evict()'s torch_gc can release its segments
            for module in to_move:
                module.to(device=device)
            self._mem_pools.pop(stem, None)

    def eval(self):
        # Put every present component on eval(); driven by the same part registry as materialize()/evict().
        # A model whose component names diverge (Wuerstchen) or that has a component outside model_parts()
        # (SD's depth_estimator, Anima's text_conditioner) overrides this.
        for part in self.model_type.model_parts():
            stem = f"{part}_1" if hasattr(self, f"{part}_1") else part
            component = getattr(self, stem)
            if component is not None:
                component.eval()

    def adapters(self) -> list[LoRAModuleWrapper]:
        # Every LoRA adapter present on a model part, in model_parts() order. Parts without a LoRA
        # (e.g. the vae, or an untrained component) contribute nothing.
        result = []
        for part in self.model_type.model_parts():
            stem = f"{part}_1" if hasattr(self, f"{part}_1") else part
            lora = getattr(self, f"{stem}_lora", None)
            if lora is not None:
                result.append(lora)
        return result

    def diffusers_to_original(self) -> list | None:
        # the canonical(diffusers) -> native key-conversion BODY (rename only) for this model's denoising
        # component. The single per-model definition, overridden by each model, reused by the four per-format
        # accessors below (LoRA original/comfy/kohya + the full-model checkpoint). None for a ships-as-diffusers
        # model where native == diffusers. Any model-specific parameter (e.g. the highest joint-block index) is
        # resolved here from the live model. Call the per-format accessors at the call sites, not this.
        return None

    def fusion_groups(self) -> list | None:
        # The model's qkv fusion groups, or None for a non-fusing model. Each entry is
        # (group_pattern, [split leaf suffixes], fused suffix, original suffix); group_pattern may contain a
        # placeholder (e.g. "transformer_blocks.{i}") matching any repeated block, or be a fixed key with no
        # placeholder for a group that occurs once. Used by the KOHYA un-flatten to collapse the live split
        # q/k/v leaves into the fused module name before the body rename, and by the full-model
        # checkpoint_diffusers_to_original pre-stage.
        return None

    # Per-format denoising bodies. Each returns a canonical(diffusers) -> native conversion for one output
    # format, defaulting to the shared diffusers_to_original() body; a model overrides only the format whose
    # native layout diverges from the others (e.g. SD's KOHYA, Z-Image's full-model COMFY). These hooks are
    # denoising-only: the structural top-prefix (strip for ORIGINAL, "diffusion_model." for COMFY, "lora_unet"
    # for KOHYA) and all text-encoder handling stay in the saver/loader mixins, driven by lora_text_encoders().

    def lora_diffusers_to_original(self) -> list | None:
        # ORIGINAL/COMFY LoRA base body (and the default for the comfy/kohya siblings). LoRA adapters are
        # trained already-fused, so the saver applies this rename forward and the loader reverses it -- no
        # fusion stage (unlike the checkpoint accessor).
        return self.diffusers_to_original()

    def lora_diffusers_to_comfy(self) -> list | None:
        # COMFY LoRA body. Defaults to the shared body so every model gets a working COMFY LoRA; a model whose
        # COMFY LoRA layout diverges from ORIGINAL overrides this.
        return self.diffusers_to_original()

    def lora_diffusers_to_kohya(self) -> list | None:
        # KOHYA LoRA body. Defaults to the shared body; a model whose KOHYA layout diverges from ORIGINAL/COMFY
        # overrides this.
        return self.diffusers_to_original()

    def checkpoint_diffusers_to_original(self) -> list | None:
        # full-model checkpoint body: an optional qkv-fusion pre-stage (fuses split q/k/v[/mlp] into the fused
        # diffusers name) then the body. A fusing model is multi-pass; a non-fusing model is the body alone; a
        # ships-as-diffusers model (no body) is None.
        body = self.diffusers_to_original()
        if body is None:
            return None
        fusion = self.fusion_groups()
        if fusion is None:
            return body
        return [qkv_fusion(fusion), body]

    def checkpoint_diffusers_to_comfy(self) -> list | None:
        # full-model COMFY checkpoint body. Unsupported by default -- only Z-Image's full-model Comfy layout
        # diverges from diffusers/original (ComfyUI #12303), so only Z-Image overrides this.
        return None

    def lora_text_encoders(self) -> list[tuple[torch.nn.Module | None, dict[ModelFormat, str]]]:
        # The model's ONLY LoRA-namespace declaration: the text encoders it trains a LoRA on, in order, each a
        # (live base module, {ModelFormat: name}) tuple. The module is handed over directly (e.g.
        # self.text_encoder_2) so the loader never guesses it from a name. The dict gives the TE's top prefix
        # per format that persists it: DIFFUSERS_LORA = the canonical / in-memory source name, KOHYA_LORA =
        # "lora_te1", COMFY_LORA = "text_encoders.clip_l.transformer". No ORIGINAL_LORA entry -- an ORIGINAL
        # file is denoising-only, and in ORIGINAL/COMFY a TE keeps its canonical name (COMFY adds the
        # text_encoders.* prefix). Empty (the default) for a transformer-only LoRA -- the common case. The
        # denoising component is not declared here; it is fully derived from the model. This is the LoRA-trained
        # subset, NOT model_parts: Flux2 has a text_encoder part but trains no TE LoRA.
        return []

    @staticmethod
    def _add_embeddings_to_prompt(
            additional_embeddings: list[BaseModelEmbedding],
            prompt: str,
    ) -> str:
        for embedding in additional_embeddings:
            prompt = prompt.replace(embedding.placeholder, embedding.joint_text_tokens)

        return prompt

    @staticmethod
    def _apply_output_embeddings(
            embeddings: list[BaseModelEmbedding],
            tokenizer: PreTrainedTokenizer,
            tokens: torch.Tensor,
            text_encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        for embedding in embeddings:
            if embedding.is_output_embedding:
                text_encoder_output = text_encoder_output.to(dtype=torch.float32)

                if embedding.joint_tokens_cache is None:
                    embedding.joint_tokens_cache = tokenizer(
                        embedding.joint_text_tokens,
                        add_special_tokens=False,
                        return_tensors="pt",
                    ).input_ids.to(text_encoder_output.device)

                batch_size = text_encoder_output.shape[0]

                embedding_tokens = embedding.joint_tokens_cache.expand(batch_size, -1)
                idx_0, idx_1, idx_2 = (tokens.unsqueeze(1) == embedding_tokens.unsqueeze(2)).nonzero(as_tuple=True)
                text_encoder_output[idx_0, idx_2] = embedding.output_vector[idx_1]

        return text_encoder_output
