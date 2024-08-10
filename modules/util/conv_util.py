from torch import nn

from modules.module.LoRAModule import LoRAModuleWrapper


def apply_circular_padding_to_conv2d(module: nn.Module | LoRAModuleWrapper):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.padding_mode = 'circular'
