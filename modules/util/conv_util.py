from modules.module.LoRAModule import LoRAModuleWrapper

from torch import nn


def apply_circular_padding_to_conv2d(module: nn.Module | LoRAModuleWrapper):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.padding_mode = 'circular'
