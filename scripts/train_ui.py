from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI
import torch

if torch.xpu.is_available():
    from ipex_to_cuda import ipex_init
    ipex_init()
    print("✓ CUDA → XPU hijacking active")

def main():
    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
