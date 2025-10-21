from collections.abc import Callable

from modules.model.BaseModel import BaseModel, TrainConfig
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.ModuleFilter import ModuleFilter

import torch


def build_muon_adam_key_fn(
    model: BaseModel,
    config: TrainConfig,
    debug_mode: bool,
) -> Callable:
    """
    Creates a function that maps a parameter to its designated optimizer type,
    'muon' or 'adam', using a configurable, layer filter.
    """
    param_map: dict[int, str] = {}

    all_processed_params: list[torch.nn.Parameter] = []

    filters: list[ModuleFilter]

    # Use user-provided patterns if they exist, otherwise use the hardcoded default.
    if config.optimizer.non_hidden_layers is not None:
        # Create filters from the user's configuration string
        patterns_list = [p.strip() for p in config.optimizer.non_hidden_layers.split(',') if p.strip()]
        filters = [ModuleFilter(p, use_regex=config.optimizer.muon_adam_regex) for p in patterns_list]
        if True:
            print(f"[MuonWithAuxAdam] Using custom non-hidden layer patterns: {patterns_list}")
    else:
        # Default list of "non-hidden" parts. These are simple substrings.
        default_patterns = [
            'time_embed', 'label_emb', 'context_embedder', 'x_embedder',
            'proj_in', 'proj_out', 'img_in', 'txt_in', 'time_text_embed',
            'norm', # Catches LayerNorm, final norms like 'norm_out', 'txt_norm', etc.
            'emb', 'embed', # General embedding layers
            'pos_encoding', 'positional_embedding',
            'dora_scale',
            'img_mod', 'txt_mod',
            'guidance_embedder',
            'ff_context',
        ]
        filters = [ModuleFilter(p, use_regex=False) for p in default_patterns]
        if True:
            print("[MuonWithAuxAdam] Using default non-hidden layer patterns.")


    def get_optim_type(param_name: str, p: torch.nn.Parameter) -> str:
        """Applies the simplified rule hierarchy to a single parameter."""
        # Rule 1: Check against the exclusion filters first.
        if any(f.matches(param_name) for f in filters):
            return 'adam'

        # Rule 2: For everything else, apply the standard μP logic.
        return 'muon' if p.ndim >= 2 else 'adam'

    # Module-based iteration & parameter mapping
    sub_modules_to_check = [
        'text_encoder', 'text_encoder_1', 'text_encoder_2', 'text_encoder_3', 'text_encoder_4',
        'unet', 'transformer',
        'text_encoder_lora', 'text_encoder_1_lora', 'text_encoder_2_lora', 'text_encoder_3_lora', 'text_encoder_4_lora',
        'unet_lora', 'transformer_lora'
    ]

    for module_prefix in sub_modules_to_check:
        module = getattr(model, module_prefix, None)
        if module is None:
            continue

        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                # For LoRA, the full name includes the original module's prefix
                full_prefix = lora_module.prefix
                for param_name, p in lora_module.named_parameters():
                    if p.requires_grad:
                        full_param_name = f"{full_prefix}.{param_name}"
                        param_map[id(p)] = get_optim_type(full_param_name, p)
                        all_processed_params.append(p)
        elif any(p.requires_grad for p in module.parameters()):
            for param_name, p in module.named_parameters():
                if p.requires_grad:
                    full_param_name = f"{module_prefix}.{param_name}"
                    param_map[id(p)] = get_optim_type(full_param_name, p)
                    all_processed_params.append(p)

    # Print a summary for verification
    if True:
        muon_params_count, adam_params_count = 0, 0
        muon_tensors, adam_tensors = 0, 0
        unassigned_params_count = 0

        for p in all_processed_params:
            optim_type = param_map.get(id(p))
            if optim_type is None:
                optim_type = 'adam'
                unassigned_params_count += 1

            if optim_type == 'muon':
                muon_params_count += p.numel()
                muon_tensors += 1
            else:
                adam_params_count += p.numel()
                adam_tensors += 1

        total_params = muon_params_count + adam_params_count
        if total_params > 0:
            muon_percent = 100 * muon_params_count / total_params
            adam_percent = 100 * adam_params_count / total_params
            print("\n--- MuonWithAuxAdam Parameter Distribution ---")
            print(f"Assigned to Muon : {muon_params_count:,} parameters ({muon_percent:.2f}%) in {muon_tensors} tensors.")
            print(f"Assigned to AdamW: {adam_params_count:,} parameters ({adam_percent:.2f}%) in {adam_tensors} tensors.")
            print(f"Total trainable  : {total_params:,} parameters")
            if unassigned_params_count > 0:
                print(f"INFO: {unassigned_params_count} trainable tensor(s) were not in checked modules and defaulted to AdamW.")

            if config.optimizer.non_hidden_layers is not None:
                unused_filters = [f._pattern for f in filters if not f.was_used()]
                if unused_filters:
                    print(f"WARNING: The following non-hidden layer patterns did not match any parameters: {unused_filters}")

            print("----------------------------------------------\n")
        else:
            print("\n[MuonWithAuxAdam] Warning: No trainable parameters found.\n")


    def layer_key_fn(p: torch.nn.Parameter) -> str:
        return param_map.get(id(p), 'adam')

    return layer_key_fn
