import re

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper

import torch


def _get_layer_key_from_name(full_name: str) -> str:
    """
    Group parameters into layers by finding the most semantically
    relevant block pattern in the parameter's name.
    """
    # Start with UNet style architectures.
    unet_indexed_patterns = [
        'down_blocks', 'up_blocks', 'input_blocks', 'output_blocks',
    ]

    # Handle singleton, non-indexed blocks.
    # We check for these with a simple string search first.
    singleton_patterns = ['mid_block', 'middle_block']
    for pattern in singleton_patterns:
        if f'.{pattern}.' in f'.{full_name}.':
            key = full_name.split(f'.{pattern}.', 1)[0] + f'.{pattern}'
            return key

    # General patterns for transformers and other models.
    general_patterns = [
        'resnets', 'attentions', 'transformer_blocks', 'resblocks',
        'blocks', 'layers', 'layer', 'experts', 'downsample'
        'double_stream_blocks', 'single_stream_blocks', 'refiner_blocks'
        'single_blocks', 'double_blocks', 'block'
        'single_transformer_blocks', 'joint_blocks', 'context_block',
        'x_block', 'block_',
    ]

    # Combine indexed patterns for a single regex pass
    all_indexed_patterns = unet_indexed_patterns + general_patterns
    regex_pattern = r'(' + '|'.join(all_indexed_patterns) + r')\.\d+'

    matches = list(re.finditer(regex_pattern, full_name))

    if not matches:
        return full_name.rpartition('.')[0]

    unet_matches = []
    for match in matches:
        pattern_word = match.group(1)
        if pattern_word in unet_indexed_patterns:
            unet_matches.append(match)

    if unet_matches:
        last_match = unet_matches[-1]
        return full_name[:last_match.end()]
    else:
        last_match = matches[-1]
        return full_name[:last_match.end()]


def build_layer_identifier_fn(model: BaseModel, debug_mode: bool) -> callable:
    """
    Creates a function that maps a parameter to its layer's string identifier
    based on its full name.
    """
    param_map = {}
    unique_layer_keys = set()

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

        count = 0
        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                full_name = lora_module.prefix.rstrip('.')
                layer_key = _get_layer_key_from_name(full_name)
                unique_layer_keys.add(layer_key)
                for _, p in lora_module.named_parameters():
                    if p.requires_grad:
                        param_map[id(p)] = layer_key
                        count += 1
        elif any(p.requires_grad for p in module.parameters()):
            for param_name, p in module.named_parameters():
                if p.requires_grad:
                    full_name = f"{module_prefix}.{param_name}"
                    layer_key = _get_layer_key_from_name(full_name)
                    unique_layer_keys.add(layer_key)
                    param_map[id(p)] = layer_key
                    count += 1

    def layer_key_fn(p: torch.nn.Parameter) -> str:
        layer_key = param_map.get(id(p))
        if layer_key is None:
            # Fallback for unmapped parameters: treat each as its own bucket.
            # This also applies to embeddings where each tensor is its own embedding (bucket).
            return id(p)
        return layer_key

    return layer_key_fn
