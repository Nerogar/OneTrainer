from torch.cuda.amp import GradScaler


class CustomGradScaler(GradScaler):

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
        # Overwrites _unscale_grads_ to always allow fp16. This is needed to train models in fp16 mode
        return super(CustomGradScaler, self)._unscale_grads_(optimizer, inv_scale, found_inf, True)
