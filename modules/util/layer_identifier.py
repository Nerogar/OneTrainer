from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.module.LoRAModule import LoRAModuleWrapper

import torch
from torch.nn import Module


def build_layer_identifier_fn(model: BaseModel, debug_mode: bool) -> callable:
    """
    Creates a function that maps a parameter to its layer's string identifier
    based on its full name. Correctly groups parameters.
    """
    if debug_mode:
        print("[Kourkoutas-β Debug] Starting to build layer key function...")

    param_map = {}

    sub_modules_to_check = {
        'text_encoder': getattr(model, 'text_encoder', None),
        'text_encoder_1': getattr(model, 'text_encoder_1', None),
        'text_encoder_2': getattr(model, 'text_encoder_2', None),
        'text_encoder_3': getattr(model, 'text_encoder_3', None),
        'text_encoder_4': getattr(model, 'text_encoder_4', None),
        'unet': getattr(model, 'unet', None),
        'transformer': getattr(model, 'transformer', None),
        'text_encoder_lora': getattr(model, 'text_encoder_lora', None),
        'text_encoder_1_lora': getattr(model, 'text_encoder_1_lora', None),
        'text_encoder_2_lora': getattr(model, 'text_encoder_2', None),
        'text_encoder_3_lora': getattr(model, 'text_encoder_3', None),
        'text_encoder_4_lora': getattr(model, 'text_encoder_4', None),
        'unet_lora': getattr(model, 'unet_lora', None),
        'transformer_lora': getattr(model, 'transformer_lora', None),
    }

    for module_name, module in sub_modules_to_check.items():
        if module is None:
            continue

        count = 0
        if isinstance(module, LoRAModuleWrapper):
            for lora_module in module.lora_modules.values():
                layer_key = lora_module.prefix.rstrip('.')
                for _param_name, p in lora_module.named_parameters():
                    if p.requires_grad:
                        param_map[id(p)] = layer_key
                        count += 1
        elif isinstance(module, torch.nn.Module):
            for param_name, p in module.named_parameters():
                if p.requires_grad:
                    # For standard modules, group by the module path
                    full_name = f"{module_name}.{param_name}"
                    param_map[id(p)] = full_name.rpartition('.')[0]
                    count += 1

        if debug_mode and count > 0:
            print(f"[Kourkoutas-β Debug] Scanned '{module_name}', found {count} trainable parameters.")

    embedding_sources = [
        'all_text_encoder_embeddings',
        'all_text_encoder_1_embeddings',
        'all_text_encoder_2_embeddings',
        'all_text_encoder_3_embeddings',
        'all_text_encoder_4_embeddings',
    ]

    count = 0
    seen_embedding_ids = set()  # Use container ID to avoid processing the same module twice

    for method_name in embedding_sources:
        if not hasattr(model, method_name):
            continue

        # Extract a suffix like '_1', '_2' to differentiate encoders
        suffix = ""
        parts = method_name.split('_')
        if len(parts) > 3 and parts[3].isdigit():
            suffix = f"_{parts[3]}"

        embeddings_list = getattr(model, method_name)()
        for i, emb_container in enumerate(embeddings_list):
            if id(emb_container) in seen_embedding_ids:
                continue
            seen_embedding_ids.add(id(emb_container))

            if isinstance(emb_container, BaseModelEmbedding) and emb_container.vector is not None and emb_container.vector.requires_grad:
                # Each embedding is its own bucket
                full_name = f"embedding{suffix}.{emb_container.placeholder}"
                param_map[id(emb_container.vector)] = full_name
                count += 1
            elif isinstance(emb_container, Module):
                for param_name, p in emb_container.named_parameters():
                    if p.requires_grad:
                        full_name = f"embedding_module{suffix}.{i}.{param_name}"
                        param_map[id(p)] = full_name.rpartition('.')[0]
                        count += 1
        if debug_mode and count > 0:
            print(f"[Kourkoutas-β Debug] Scanned embeddings, found {count} trainable embedding vectors.")

    if debug_mode:
        print(f"[Kourkoutas-β Debug] Total trainable parameters mapped: {len(param_map)}")
        if len(param_map) > 0:
            print("[Kourkoutas-β Debug] All mapped parameter names:")
            for name in param_map.values():
                print(f"  - Mapped to bucket key: '{name}'")

    def layer_key_fn(p: torch.nn.Parameter) -> str:
        layer_key = param_map.get(id(p))
        if layer_key is None:
            return 'unmapped_param'
        return layer_key

    return layer_key_fn
