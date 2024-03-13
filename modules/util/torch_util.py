import gc

import torch


def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
