from collections.abc import Callable

from modules.model.BaseModel import BaseModel, TrainConfig
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType
from modules.util.ModuleFilter import ModuleFilter

import torch


def build_muon_adam_key_fn(
    model: BaseModel,
    config: TrainConfig,
) -> Callable:
    """
    Creates a function that maps a parameter to its designated optimizer type,
    'muon' or 'adam', using a configurable, layer filter.
    """
    param_map: dict[int, str] = {}

    all_processed_params: list[torch.nn.Parameter] = []

    filters: list[ModuleFilter]

    # Use user-provided patterns if they exist, otherwise use the hardcoded default.
    if config.optimizer.muon_hidden_layers is not None:
        # Create filters from the user's configuration string
        patterns_list = [p.strip() for p in config.optimizer.muon_hidden_layers.split(',') if p.strip()]
        filters = [ModuleFilter(p, use_regex=config.optimizer.muon_adam_regex) for p in patterns_list]
        if True:
            print(f"[MuonWithAuxAdam] Using custom hidden layer patterns: {patterns_list}")
    else:
        # Default list of "hidden" parts.
        match model.model_type:
            case ModelType.STABLE_DIFFUSION_15 | ModelType.STABLE_DIFFUSION_15_INPAINTING | ModelType.STABLE_DIFFUSION_20_BASE | ModelType.STABLE_DIFFUSION_20_INPAINTING | ModelType.STABLE_DIFFUSION_20 | ModelType.STABLE_DIFFUSION_21 | ModelType.STABLE_DIFFUSION_21_BASE | ModelType.STABLE_DIFFUSION_XL_10_BASE | ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING | ModelType.STABLE_CASCADE_1 | ModelType.WUERSTCHEN_2:
                default_patterns = [
                    'block', # UNet
                    'text_model.encoder.layers', # TEs (CLIPs)
                ]
            case ModelType.STABLE_DIFFUSION_3 | ModelType.STABLE_DIFFUSION_35 | ModelType.SANA  | ModelType.FLUX_DEV_1  | ModelType.CHROMA_1  | ModelType.QWEN  | ModelType.PIXART_ALPHA | ModelType.PIXART_SIGMA:
                default_patterns = [
                    'transformer_blocks',
                    'encoder.block', # TE (T5)
                ]
            case ModelType.HI_DREAM_FULL:
                default_patterns = [
                    'caption_projection',
                    'double_stream_blocks',
                    'single_stream_blocks',
                ]
            case _: # Unmatched cases
                raise NotImplementedError(f"Default hidden layer patterns are not defined for model type: {model.model_type}")
        filters = [ModuleFilter(p, use_regex=False) for p in default_patterns]
        if True:
            print(f"[MuonWithAuxAdam] Using default hidden layer patterns for {model.model_type}.")


    def get_optim_type(param_name: str, p: torch.nn.Parameter) -> str:
        """Applies the simplified rule hierarchy to a single parameter."""
        # Rule 1: Check against the exclusion filters first.
        if any(f.matches(param_name) for f in filters) and len(p.shape) != 1:
            return 'muon'

        # Rule 2: For everything else, use Adam
        return 'adam'

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

            if config.optimizer.muon_hidden_layers is not None:
                unused_filters = [f._pattern for f in filters if not f.was_used()]
                if unused_filters:
                    print(f"WARNING: The following hidden layer patterns did not match any parameters: {unused_filters}")

            print("----------------------------------------------\n")
        else:
            print("\n[MuonWithAuxAdam] Warning: No trainable parameters found.\n")


    def layer_key_fn(p: torch.nn.Parameter) -> str:
        return param_map.get(id(p), 'adam')

    return layer_key_fn

def split_parameters_for_muon(
    parameters: list[dict],
    layer_key_fn: Callable,
    config: TrainConfig,
) -> tuple[list[dict], bool]:
    """
    Splits parameter groups into 'muon' and 'adam' subgroups for MuonWithAuxAdam.
    If MuonWithAuxAdam is not active, returns the original parameters.
    """
    optimizer_config = config.optimizer

    has_adam_params = False
    if layer_key_fn:
        for group in parameters:
            for p in group['params']:
                if p.requires_grad and layer_key_fn(p) == 'adam':
                    has_adam_params = True
                    break
            if has_adam_params:
                break

    MuonWithAuxAdam = optimizer_config.MuonWithAuxAdam and has_adam_params

    # If not using AuxAdam, just use the original parameter groups
    if not (MuonWithAuxAdam and layer_key_fn):
        return parameters, MuonWithAuxAdam

    final_param_groups = []
    for group in parameters:
        muon_params = [p for p in group['params'] if p.requires_grad and layer_key_fn(p) == 'muon']
        adam_params = [p for p in group['params'] if p.requires_grad and layer_key_fn(p) == 'adam']

        if muon_params:
            muon_group = group.copy()
            muon_group['params'] = muon_params
            muon_group['optim_type'] = 'muon'
            final_param_groups.append(muon_group)

        if adam_params:
            adam_group = group.copy()
            adam_group['params'] = adam_params
            adam_group['optim_type'] = 'adam'
            # Set Adam-specific LR
            base_adam_lr = optimizer_config.muon_adam_lr if optimizer_config.muon_adam_lr is not None else config.learning_rate
            te1_adam_lr = optimizer_config.muon_te1_adam_lr
            te2_adam_lr = optimizer_config.muon_te2_adam_lr
            adam_lr = base_adam_lr
            original_name = group.get('name')
            if original_name in ('text_encoder', 'text_encoder_1', 'text_encoder_lora', 'text_encoder_1_lora'):
                adam_lr = te1_adam_lr if te1_adam_lr is not None else base_adam_lr
            if original_name in ('text_encoder_2', 'text_encoder_2_lora'):
                adam_lr = te2_adam_lr if te2_adam_lr is not None else base_adam_lr
            adam_group['lr'] = adam_lr
            adam_group['initial_lr'] = adam_lr
            final_param_groups.append(adam_group)

    return final_param_groups, MuonWithAuxAdam
