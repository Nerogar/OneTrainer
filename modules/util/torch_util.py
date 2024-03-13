import gc

import torch


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
