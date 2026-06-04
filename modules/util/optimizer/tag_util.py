from modules.module.LoRAModule import LoRAModuleWrapper

import torch


def tag_peft_parameters(model: torch.nn.Module | None):
    """
    Tags PEFT parameters with attributes like `_is_lora_A`, `_is_lora_B`,
    `_is_oft`, and `_is_dora_scale` based on their names.
    This is required to apply correct scaling and optimizations for certain features in adv_optm library.
    """
    def apply_tags(name: str, p: torch.nn.Parameter):
        if name.endswith(("lora_down.weight", "lokr_w1_b", "lokr_w2_b")):
            # Down projection
            p._is_lora_A = True
        elif name.endswith(("lora_up.weight", "lokr_w1_a", "lokr_w2_a")):
            # Up projection
            p._is_lora_B = True
        elif name.endswith(("dora_scale", "dora_log_multiplier")):
            # Vector in shape of >= 2D tensor
            p._is_dora_scale = True
        elif name.endswith("oft_R.weight"):
            # Set of independent vectors (rank, n_elements)
            p._is_oft = True

    for module in vars(model).values():
        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                for param_name, p in lora_module.named_parameters():
                    apply_tags(param_name, p)
        elif isinstance(module, torch.nn.Module):
            for param_name, p in module.named_parameters():
                apply_tags(param_name, p)
