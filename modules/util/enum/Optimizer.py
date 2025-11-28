from modules.util.enum.BaseEnum import BaseEnum

import torch


class Optimizer(BaseEnum):
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

    #Pytorch Optimizers
    ADABELIEF = 'ADABELIEF'
    TIGER = 'TIGER'
    AIDA = 'AIDA'
    YOGI = 'YOGI'

    def pretty_print(self):
        return {
            Optimizer.ADAGRAD: "AdaGrad",
            Optimizer.ADAGRAD_8BIT: "AdaGrad 8 bit",
            Optimizer.ADAM: "Adam",
            Optimizer.ADAM_8BIT: "Adam 8 bit",
            Optimizer.ADAMW: "AdamW",
            Optimizer.ADAMW_8BIT: "AdamW 8 bit",
            Optimizer.ADAMW_ADV: "AdamW Advanced",
            Optimizer.AdEMAMix: "AdEMAMix",
            Optimizer.AdEMAMix_8BIT: "AdEMAMix 8 bit",
            Optimizer.SIMPLIFIED_AdEMAMix: "Simplified AdEMAMix",
            Optimizer.ADOPT: "ADOPT",
            Optimizer.ADOPT_ADV: "ADOPT Advanced",
            Optimizer.LAMB: "LAMB",
            Optimizer.LAMB_8BIT: "LAMB 8 bit",
            Optimizer.LARS: "LARS",
            Optimizer.LARS_8BIT: "LARS 8 bit",
            Optimizer.LION: "Lion",
            Optimizer.LION_8BIT: "Lion 8 bit",
            Optimizer.LION_ADV: "Lion Advanced",
            Optimizer.RMSPROP: "RMSProp",
            Optimizer.RMSPROP_8BIT: "RMSProp 8 bit",
            Optimizer.SGD: "SGD",
            Optimizer.SGD_8BIT: "SGD 8 bit",
            Optimizer.SCHEDULE_FREE_ADAMW: "Schedule Free AdamW",
            Optimizer.SCHEDULE_FREE_SGD: "Schedule Free SGD",
            Optimizer.DADAPT_ADA_GRAD: "DAdapt AdaGrad",
            Optimizer.DADAPT_ADAM: "DAdapt Adam",
            Optimizer.DADAPT_ADAN: "DAdapt ADAN",
            Optimizer.DADAPT_LION: "DAdapt Lion",
            Optimizer.DADAPT_SGD: "DAdapt SGD",
            Optimizer.PRODIGY: "Prodigy",
            Optimizer.PRODIGY_PLUS_SCHEDULE_FREE: "Prodigy Plus Schedule Free",
            Optimizer.PRODIGY_ADV: "Prodigy Advanced",
            Optimizer.LION_PRODIGY_ADV: "Lion Prodigy Advanced",
            Optimizer.ADAFACTOR: "Adafactor",
            Optimizer.CAME: "CAME",
            Optimizer.CAME_8BIT: "CAME 8 bit",
            Optimizer.ADABELIEF: "AdaBelief",
            Optimizer.TIGER: "Tiger",
            Optimizer.AIDA: "Aida",
            Optimizer.YOGI: "Yogi",
        }[self]

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
