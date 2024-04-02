import torch
from torch._prims_common import DeviceLikeType


from modules.util.config.TrainConfig import TrainConfig


def is_zluda(device: DeviceLikeType):
    device = torch.device(device)
    if device.type == "cpu":
        return False
    return torch.cuda.get_device_name(device).endswith("[ZLUDA]")


def test(device: DeviceLikeType):
    device = torch.device(device)
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        return out.sum().is_nonzero()
    except Exception:
        return False


def initialize():
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


def initialize_devices(config: TrainConfig):
    if not is_zluda(config.train_device) and not is_zluda(config.temp_device):
        return
    devices = [config.train_device, config.temp_device,]
    for i in range(2):
        device = torch.device(devices[i])
        if not test(devices[i]):
            print(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            devices[i] = 'cpu'
    config.train_device, config.temp_device = devices
