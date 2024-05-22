from typing import Union
import torch
from torch._prims_common import DeviceLikeType
from modules.util.config.TrainConfig import TrainConfig


def is_zluda(device: DeviceLikeType):
    device = torch.device(device)
    if device.type == "cpu":
        return False
    return torch.cuda.get_device_name(device).endswith("[ZLUDA]")


def test(device: DeviceLikeType) -> Union[Exception, None]:
    device = torch.device(device)
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        assert out.sum().is_nonzero()
        return None
    except Exception as e:
        return e


def initialize():
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def initialize_devices(config: TrainConfig):
    if not is_zluda(config.train_device) and not is_zluda(config.temp_device):
        return
    devices = [config.train_device, config.temp_device,]
    for i in range(2):
        device = torch.device(devices[i])
        result = test(device)
        if result is not None:
            print(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            print(result)
            devices[i] = 'cpu'
    config.train_device, config.temp_device = devices
