import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

from modules.model.BaseModel import BaseModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert_lora_util import (
    convert_lora_suffix_ab,
    convert_to_diffusers,
    kohya_flatten,
    lora_kohya_conversion,
    lora_original_conversion,
)
from modules.util.convert_util import convert, remove_prefix
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class LoRASaverMixin(
    DtypeModelSaverMixin,
    metaclass=ABCMeta,
):
    def __init__(self):
        super().__init__()

    def save(
            self,
            model: BaseModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        self._save(model, output_model_format, output_model_destination, dtype)

    @abstractmethod
    def _get_state_dict(
            self,
            model: BaseModel,
    ) -> dict[str, Tensor]:
        pass

    def _write_lora_file(
            self,
            model: BaseModel,
            destination: str,
            save_state_dict: dict[str, Tensor],
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    # Per-format save methods. Every model is migrated to the convert() tool. KOHYA and ORIGINAL/COMFY are
    # derived from the model here (a saver overrides only when it diverges -- StableDiffusion's KOHYA); only
    # LEGACY, the frozen per-model historical layout, is always per-model. DIFFUSERS is key-set-free
    # (canonical == diffusers namespace) and identical for every model.

    def _save_diffusers(self, model: BaseModel, destination: str, dtype: torch.dtype | None):
        # DIFFUSERS format: HF diffusers convention (not the peft library's native layout).
        # Key-set-free (canonical == diffusers namespace).
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        save_state_dict = convert_to_diffusers(save_state_dict)
        self._write_lora_file(model, destination, save_state_dict)

    def _convert_kohya(self, model: BaseModel, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # KOHYA = kohya-ss / musubi-tuner: the denoising model (fused qkv where the model fuses) under the
        # universal lora_unet_ prefix, each text encoder under its lora_te* prefix, then flattened. Fully
        # derived from the model (lora_kohya_conversion, whose denoising body is model.lora_diffusers_to_kohya()
        # -- SD's KOHYA divergence lives in that model hook, not a saver override). SD reuses this for its
        # byte-identical LEGACY (StableDiffusionLoRASaver._convert_legacy).
        state_dict = convert(state_dict, lora_kohya_conversion(model), strict=True)
        return kohya_flatten(state_dict)

    def _save_kohya(self, model: BaseModel, destination: str, dtype: torch.dtype | None):
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        save_state_dict = self._convert_kohya(model, save_state_dict)
        self._write_lora_file(model, destination, save_state_dict)

    def _convert_legacy(self, model: BaseModel, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        # LEGACY = OneTrainer's historical per-model output. The default refuses it: a model opts in only by
        # overriding this with its historical layout. Models whose only historical LoRA output was a
        # never-loadable format (HiDream's OMI, Sana / Wuerstchen v2's dotted) inherit this raise, as does
        # any model that never had a LEGACY output. Kept in sync with ModelType.supported_lora_formats,
        # which drops LEGACY_LORA from the UI for exactly these models.
        raise NotImplementedError(
            f"The LEGACY LoRA output format is not supported for {model.model_type}.")

    def _save_legacy(self, model: BaseModel, destination: str, dtype: torch.dtype | None):
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        save_state_dict = self._convert_legacy(model, save_state_dict)
        self._write_lora_file(model, destination, save_state_dict)

    def _save_original(self, model: BaseModel, destination: str, dtype: torch.dtype | None):
        state_dict = self._get_state_dict(model)
        # ORIGINAL puts the denoising model at the top level with no prefix, which leaves no namespace for a
        # second component: a trained text encoder (or bundled embeddings) would have to sit under its own
        # prefix alongside the unprefixed denoising keys, an asymmetric layout no external tool reads. Refuse
        # it -- COMFY (every component under its own prefix, the denoising model as "diffusion_model.") is the
        # format for a multi-component LoRA. Components are the distinct top-level key segments (transformer /
        # text_encoder[_n] / bundle_emb).
        components = {key.split(".", 1)[0] for key in state_dict}
        if len(components) > 1:
            raise RuntimeError(
                "The ORIGINAL LoRA format places the denoising model at the top level with no prefix and "
                f"cannot represent more than one trained component (got {len(components)}: "
                f"{', '.join(sorted(components))}). Use the COMFY format for a multi-component LoRA.")

        conversion = lora_original_conversion(model, model.lora_diffusers_to_original())
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        save_state_dict = convert(save_state_dict, conversion, strict=True)
        # drop the denoising component's top prefix (denoising model only -- TEs keep original names;
        # strict=False so the TE clip_l./t5. and bundle_emb. keys pass through untouched).
        save_state_dict = convert(save_state_dict, remove_prefix(model.model_type.denoising_model_part()), strict=False)
        # ORIGINAL suffix = lora_A/lora_B, alpha and dora_scale kept (peft_convention=False).
        save_state_dict = convert_lora_suffix_ab(save_state_dict, peft_convention=False)
        self._write_lora_file(model, destination, save_state_dict)

    def _save_comfy(self, model: BaseModel, destination: str, dtype: torch.dtype | None):
        conversion = lora_original_conversion(model, model.lora_diffusers_to_comfy())
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        save_state_dict = convert(save_state_dict, conversion, strict=True)
        # COMFY needs a Comfy-native "text_encoders.<prefix>" name for every trained text encoder, or Comfy
        # silently drops those keys on load. Refuse to write a half-readable file: every TE component present
        # (top-level segment that is neither the denoising model nor bundle_emb) must have a prefix. Models
        # Comfy can't load (Sana, Wuerstchen) and any future model whose lora_text_encoders declares a TE with
        # no COMFY_LORA name raise here instead. The canonical -> COMFY_LORA map is pulled from the declaration.
        te_prefixes = {te_names[ModelFormat.DIFFUSERS_LORA]: te_names[ModelFormat.COMFY_LORA]
                       for _te_module, te_names in model.lora_text_encoders() if ModelFormat.COMFY_LORA in te_names}
        te_components = {key.split(".", 1)[0] for key in save_state_dict} - {model.model_type.denoising_model_part(), "bundle_emb"}
        missing = te_components - te_prefixes.keys()
        if missing:
            raise RuntimeError(
                f"The COMFY LoRA format has no Comfy-native text-encoder mapping for {', '.join(sorted(missing))} "
                "on this model, so ComfyUI would silently drop those keys on load.")
        # swap the denoising component's top prefix for Comfy's "diffusion_model." (denoising model only
        # -- TEs keep their canonical names; strict=False so the TE and bundle_emb. keys pass through untouched).
        save_state_dict = convert(save_state_dict, [(model.model_type.denoising_model_part(), "diffusion_model")], strict=False)
        # rename each text encoder's canonical name to Comfy's native "text_encoders.<prefix>" (its COMFY_LORA
        # declaration); strict=False so diffusion_model. and bundle_emb. pass through.
        if te_prefixes:
            save_state_dict = convert(save_state_dict, list(te_prefixes.items()), strict=False)
        # COMFY suffix = lora_A/lora_B, alpha and dora_scale kept (Comfy honors them).
        save_state_dict = convert_lora_suffix_ab(save_state_dict, peft_convention=False)
        self._write_lora_file(model, destination, save_state_dict)

    def _save_internal(self, model: BaseModel, destination: str):
        # INTERNAL resume snapshot: the raw canonical in-memory dict (no conversion). OMI (the old
        # snapshot namespace for not-yet-migrated models) has been dropped.
        os.makedirs(destination, exist_ok=True)
        state_dict = self._get_state_dict(model)
        save_state_dict = self._convert_state_dict_dtype(state_dict, None)
        self._write_lora_file(model, os.path.join(destination, "lora", "lora.safetensors"), save_state_dict)

    def _save(
            self,
            model: BaseModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS_LORA:
                self._save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.KOHYA_LORA:
                self._save_kohya(model, output_model_destination, dtype)
            case ModelFormat.LEGACY_LORA:
                self._save_legacy(model, output_model_destination, dtype)
            case ModelFormat.ORIGINAL_LORA:
                self._save_original(model, output_model_destination, dtype)
            case ModelFormat.COMFY_LORA:
                self._save_comfy(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self._save_internal(model, output_model_destination)
            case _:
                raise NotImplementedError(f"Unsupported LoRA output format: {output_model_format}")
