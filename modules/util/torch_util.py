import gc

import torch


def torch_gc():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
