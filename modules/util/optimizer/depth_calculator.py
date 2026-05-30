import re

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType

import torch


def calculate_n_layers(model: BaseModel) -> dict[str, int]:
    """
    Calculates the number of residual layers (the depth) in each component of the model.
    """
    match model.model_type:
        case (ModelType.STABLE_DIFFUSION_15 | ModelType.STABLE_DIFFUSION_15_INPAINTING |
              ModelType.STABLE_DIFFUSION_20_BASE | ModelType.STABLE_DIFFUSION_20_INPAINTING |
              ModelType.STABLE_DIFFUSION_20 | ModelType.STABLE_DIFFUSION_21 |
              ModelType.STABLE_DIFFUSION_21_BASE | ModelType.STABLE_DIFFUSION_XL_10_BASE |
              ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING | ModelType.STABLE_CASCADE_1 |
              ModelType.WUERSTCHEN_2):
            default_patterns = ['transformer_blocks', 'resnets', 'layers']
        case (ModelType.STABLE_DIFFUSION_3 | ModelType.STABLE_DIFFUSION_35 | ModelType.SANA |
              ModelType.FLUX_DEV_1 | ModelType.FLUX_2 | ModelType.CHROMA_1 | ModelType.QWEN |
              ModelType.PIXART_ALPHA | ModelType.PIXART_SIGMA):
            default_patterns = ['transformer_blocks', 'single_transformer_blocks', 'encoder.block']
        case ModelType.HI_DREAM_FULL:
            default_patterns = ['double_stream_blocks', 'single_stream_blocks']
        case ModelType.Z_IMAGE:
            default_patterns = [
                'layers',
                'refiner',
            ]
        case _:
            raise NotImplementedError(f"Scaled Optimizer is not implemented for model type: {model.model_type}")

    # Build the regex pattern
    joined_patterns = "|".join([re.escape(p) for p in default_patterns])
    pattern = re.compile(rf'(?:^|\.)(?:{joined_patterns})\.\d+$')

    layer_counts = {}

    # Iterate over model components (e.g., 'unet', 'text_encoder', 'transformer')
    for attr_name, module in vars(model).items():
        # Identify the 'Ground Truth' blocks in this component.
        target_module = module
        if isinstance(module, LoRAModuleWrapper):
            target_module = module.orig_module
        valid_component_blocks = set()
        if isinstance(target_module, torch.nn.Module):
            for name, _ in target_module.named_modules():
                if pattern.search(name):
                    valid_component_blocks.add(name)
        if not valid_component_blocks:
            continue
        count = len(valid_component_blocks)
        if count > 0:
            layer_counts[attr_name] = count

    return layer_counts


def inject_depth_into_param_groups(model: BaseModel, parameters):
    """
    Calculates the model depth and injects the 'n_layers' key directly
    into the optimizer parameter groups.
    """
    n_layers_map = calculate_n_layers(model)

    for group in parameters:
        group_name = group.get('name')
        if group_name in n_layers_map:
            group['n_layers'] = n_layers_map[group_name]
        else:
            group['n_layers'] = n_layers_map.get('default', 1)
