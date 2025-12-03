from enum import Enum

import torch


class Optimizer(Enum):
    # Sorted by origin (BNB / torch first, then DADAPT), then by adapter name, then interleaved by variant.

    # BNB Standard & 8-bit
    ADAGRAD = 'ADAGRAD'
    ADAGRAD_8BIT = 'ADAGRAD_8BIT'

    # 32 bit is torch and not bnb
    ADAM = 'ADAM'
    ADAM_8BIT = 'ADAM_8BIT'

    # 32 bit is torch and not bnb
    ADAMW = 'ADAMW'
    ADAMW_8BIT = 'ADAMW_8BIT'
    ADAMW_ADV = 'ADAMW_ADV'

    AdEMAMix = 'AdEMAMix'
    AdEMAMix_8BIT = "AdEMAMix_8BIT"
    SIMPLIFIED_AdEMAMix = "SIMPLIFIED_AdEMAMix"

    ADOPT = 'ADOPT'
    ADOPT_ADV = 'ADOPT_ADV'

    LAMB = 'LAMB'
    LAMB_8BIT = 'LAMB_8BIT'

    LARS = 'LARS'
    LARS_8BIT = 'LARS_8BIT'

    LION = 'LION'
    LION_8BIT = 'LION_8BIT'
    LION_ADV = 'LION_ADV'

    RMSPROP = 'RMSPROP'
    RMSPROP_8BIT = 'RMSPROP_8BIT'

    # 32 bit is torch and not bnb
    SGD = 'SGD'
    SGD_8BIT = 'SGD_8BIT'

    # Schedule-free optimizers
    SCHEDULE_FREE_ADAMW = 'SCHEDULE_FREE_ADAMW'
    SCHEDULE_FREE_SGD = 'SCHEDULE_FREE_SGD'

    # DADAPT
    DADAPT_ADA_GRAD = 'DADAPT_ADA_GRAD'
    DADAPT_ADAM = 'DADAPT_ADAM'
    DADAPT_ADAN = 'DADAPT_ADAN'
    DADAPT_LION = 'DADAPT_LION'
    DADAPT_SGD = 'DADAPT_SGD'

    # Prodigy
    PRODIGY = 'PRODIGY'
    PRODIGY_PLUS_SCHEDULE_FREE = 'PRODIGY_PLUS_SCHEDULE_FREE'
    PRODIGY_ADV = 'PRODIGY_ADV'
    LION_PRODIGY_ADV = 'LION_PRODIGY_ADV'

    # ADAFACTOR
    ADAFACTOR = 'ADAFACTOR'

    # CAME
    CAME = 'CAME'
    CAME_8BIT = 'CAME_8BIT'

    # MUON
    MUON = 'MUON'
    MUON_ADV = 'MUON_ADV'
    ADAMUON_ADV = 'ADAMUON_ADV'

    #Pytorch Optimizers
    ADABELIEF = 'ADABELIEF'
    TIGER = 'TIGER'
    AIDA = 'AIDA'
    YOGI = 'YOGI'

    @property
    def is_adaptive(self):
        return self in [
            self.DADAPT_SGD,
            self.DADAPT_ADAM,
            self.DADAPT_ADAN,
            self.DADAPT_ADA_GRAD,
            self.DADAPT_LION,
            self.PRODIGY,
            self.PRODIGY_PLUS_SCHEDULE_FREE,
            self.PRODIGY_ADV,
            self.LION_PRODIGY_ADV,
        ]

    @property
    def is_schedule_free(self):
        return self in [
            self.SCHEDULE_FREE_ADAMW,
            self.SCHEDULE_FREE_SGD,
            self.PRODIGY_PLUS_SCHEDULE_FREE,
        ]

    def supports_fused_back_pass(self):
        return self in [
            Optimizer.ADAFACTOR,
            Optimizer.CAME,
            Optimizer.CAME_8BIT,
            Optimizer.ADAM,
            Optimizer.ADAMW,
            Optimizer.ADAMW_ADV,
            Optimizer.ADOPT_ADV,
            Optimizer.SIMPLIFIED_AdEMAMix,
            Optimizer.PRODIGY_PLUS_SCHEDULE_FREE,
            Optimizer.PRODIGY_ADV,
            Optimizer.LION_ADV,
            Optimizer.LION_PRODIGY_ADV,
            Optimizer.MUON_ADV,
            Optimizer.ADAMUON_ADV,
        ]

    # Small helper for adjusting learning rates to adaptive optimizers.
    def maybe_adjust_lrs(self, lrs: dict[str, float], optimizer: torch.optim.Optimizer):
        if self.is_adaptive:
            return {
                # Return `effective_lr * d` if "effective_lr" key present, otherwise return `lr * d`
                key: (optimizer.param_groups[i].get("effective_lr", lr) * optimizer.param_groups[i].get("d", 1.0)
                      if lr is not None else None)
                for i, (key, lr) in enumerate(lrs.items())
            }
        return lrs

    def __str__(self):
        return self.value
