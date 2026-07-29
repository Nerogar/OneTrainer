from diffusers import Krea2Transformer2DModel

from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.krea2.Krea2LoRASaver import Krea2LoRASaver
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ModuleFilter import ModuleFilter

import torch
from torch import nn


class Krea2FineTuneLoRAExporter:
    """Compress a Krea2 full fine-tune delta directly into a standard LoRA file."""

    def export(self, model: Krea2Model, config: TrainConfig, destination: str):
        rank = config.full_fine_tune_lora_rank
        if rank < 1:
            raise ValueError("Full fine-tune LoRA export rank must be at least 1")

        # Load only the original transformer during export. This avoids a second
        # model copy occupying VRAM or producing an intermediate full checkpoint.
        base = Krea2Transformer2DModel.from_pretrained(
            config.base_model_name, subfolder="transformer", torch_dtype=torch.float32,
        )
        base_modules = dict(base.named_modules())
        filters = [ModuleFilter(pattern, use_regex=config.layer_filter_regex)
                   for pattern in config.layer_filter.split(",") if pattern]
        state_dict: dict[str, torch.Tensor] = {}
        try:
            for name, trained_module in model.transformer.named_modules():
                if not isinstance(trained_module, nn.Linear):
                    continue
                if filters and not any(module_filter.matches(name) for module_filter in filters):
                    continue
                base_module = base_modules.get(name)
                if not isinstance(base_module, nn.Linear) or base_module.weight.shape != trained_module.weight.shape:
                    raise RuntimeError(f"Base Krea2 transformer layer does not match trained layer: {name}")

                delta = trained_module.weight.detach().to(device="cpu", dtype=torch.float32, copy=True)
                delta.sub_(base_module.weight.detach().to(dtype=torch.float32))
                layer_rank = min(rank, min(delta.shape))
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                root_s = s[:layer_rank].sqrt()
                prefix = f"transformer.{name}"
                state_dict[f"{prefix}.lora_down.weight"] = root_s[:, None] * vh[:layer_rank, :]
                state_dict[f"{prefix}.lora_up.weight"] = u[:, :layer_rank] * root_s[None, :]
                # alpha == rank makes up @ down the truncated-SVD delta.
                state_dict[f"{prefix}.alpha"] = torch.tensor(float(layer_rank))
        finally:
            del base

        if not state_dict:
            raise RuntimeError("No Krea2 linear layers matched the LoRA export filter")
        Krea2LoRASaver().save_state_dict(
            model, state_dict, config.full_fine_tune_lora_format, destination, config.output_dtype.torch_dtype(),
        )
