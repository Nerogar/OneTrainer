import os
import traceback
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.load_lora_util import (
    lora_module_names,
    normalize_various,
    to_canonical,
)
from modules.util.ModelNames import ModelNames

import torch

from safetensors.torch import load_file


class LoRALoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    # Per-model load knowledge, mirroring the saver's hooks. The per-format denoising conversions are owned by
    # the model (model.lora_diffusers_to_{original,comfy,kohya}() / model.fusion_groups()) and read straight in
    # _to_canonical; a ships-as-diffusers model leaves them at their None default, so no per-loader override is
    # needed. SD's KOHYA divergence (diffusers UNet names while ORIGINAL/COMFY are sgm) lives in its
    # lora_diffusers_to_kohya() model hook, not a loader override.

    def _denoising_module(self, model: BaseModel) -> object:
        # the live denoising base module, used by the KOHYA/LEGACY un-flatten to enumerate native module
        # names. Defaults to the attribute named by the denoising model part, but the canonical prefix need
        # not equal the attribute name (Wuerstchen: component "prior" but attribute model.prior_prior), so the
        # model overrides this when they differ -- mirroring the live-module element of the TE hook.
        return getattr(model, model.model_type.denoising_model_part())

    def _fusion_groups(self, model: BaseModel) -> list | None:
        # Only the KOHYA load path consults the fusion groups (to un-flatten the on-disk fused keys); the
        # other families carry the file's fusion state in their key names.
        return model.fusion_groups()

    def _legacy_conversion(self, model: BaseModel) -> list | None:
        # Canonical->LEGACY conversion (mirrors the saver's _convert_legacy), one entry per component: the
        # denoising component flattened under lora_<component> (DIFFUSERS body, split/unfused -> None), each
        # text encoder under lora_te<n> (n by order), bundle_emb passthrough -- the saver's mixture namespace
        # restricted to this model's components. Overrides: PixArt/Chroma use a no-digit lora_te; Stable
        # Cascade uses lora_prior_unet + a native split-attn body. Sana / Wuerstchen v2 return None so a
        # LEGACY file raises instead of silently mis-loading.
        component = model.model_type.denoising_model_part()
        conversion = [(component, f"lora_{component}")]
        for i, (_te_module, te_names) in enumerate(model.lora_text_encoders(), start=1):
            conversion.append((te_names[ModelFormat.DIFFUSERS_LORA], f"lora_te{i}"))
        conversion.append(("bundle_emb", "bundle_emb"))
        return conversion

    def _to_canonical(self, model: BaseModel, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Layer 1 (model-independent) then Layer 2 (the shared, model-independent namespace dispatch). All
        # per-model knowledge is gathered from the model here and handed to to_canonical as plain data, so
        # the dispatch itself lives in exactly one place (load_lora_util.to_canonical), shared with offline
        # callers (the LoRA stress harness) that supply the same data from a captured key list.
        state_dict = normalize_various(state_dict)

        component = model.model_type.denoising_model_part()
        # a TE is declared as a (live base module, {ModelFormat: name} dict) tuple; its canonical prefix is the
        # in-memory source name (ModelFormat.DIFFUSERS_LORA) and its live base module is the module the model
        # handed us directly (no attribute-name guessing). The native name in ORIGINAL/COMFY is the canonical
        # name (the bare native name is never persisted), so the loader carries only (canonical, kohya).
        text_encoders = model.lora_text_encoders()
        module_names = {component: lora_module_names(self._denoising_module(model))}
        module_names.update({
            te_names[ModelFormat.DIFFUSERS_LORA]: lora_module_names(te_module)
            for te_module, te_names in text_encoders})
        return to_canonical(
            state_dict,
            component=component,
            original_conversion=model.lora_diffusers_to_original(),
            comfy_conversion=model.lora_diffusers_to_comfy(),
            fusion_groups=self._fusion_groups(model),
            text_encoders=[(te_names[ModelFormat.DIFFUSERS_LORA], te_names[ModelFormat.KOHYA_LORA])
                           for _te_module, te_names in text_encoders],
            legacy_conversion=self._legacy_conversion(model),
            kohya_conversion=model.lora_diffusers_to_kohya(),
            module_names=module_names,
            # the COMFY-native prefix each TE takes, pulled straight from its declaration (canonical ->
            # COMFY_LORA); to_canonical stays decoupled from the model and takes this plain dict, the same one
            # the offline stress harness builds from a captured key list. Omits TEs with no COMFY_LORA name.
            comfy_te_prefixes={te_names[ModelFormat.DIFFUSERS_LORA]: te_names[ModelFormat.COMFY_LORA]
                               for _te_module, te_names in text_encoders if ModelFormat.COMFY_LORA in te_names},
        )

    def __load_safetensors(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        state_dict = load_file(lora_name)
        model.lora_state_dict = self._to_canonical(model, state_dict)

    def __load_ckpt(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        state_dict = torch.load(lora_name, weights_only=True)
        model.lora_state_dict = self._to_canonical(model, state_dict)

    def __load_internal(
            self,
            model: BaseModel,
            lora_name: str,
    ):
        if os.path.exists(os.path.join(lora_name, "meta.json")):
            safetensors_lora_name = os.path.join(lora_name, "lora", "lora.safetensors")
            if os.path.exists(safetensors_lora_name):
                self.__load_safetensors(model, safetensors_lora_name)
        else:
            raise Exception("not an internal model")

    def _load(
            self,
            model: BaseModel,
            model_names: ModelNames,
    ):
        stacktraces = []

        if model_names.lora == "":
            return

        try:
            self.__load_internal(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        if model_names.lora.endswith(".ckpt"):
            try:
                self.__load_ckpt(model, model_names.lora)
                return
            except Exception:
                stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(model, model_names.lora)
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
