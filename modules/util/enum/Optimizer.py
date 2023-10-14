from enum import Enum


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

    LAMB = 'LAMB'
    LAMB_8BIT = 'LAMB_8BIT'

    LARS = 'LARS'
    LARS_8BIT = 'LARS_8BIT'

    LION = 'LION'
    LION_8BIT = 'LION_8BIT'

    RMSPROP = 'RMSPROP'
    RMSPROP_8BIT = 'RMSPROP_8BIT'

    # 32 bit is torch and not bnb
    SGD = 'SGD'
    SGD_8BIT = 'SGD_8BIT'

    # DADAPT
    DADAPT_ADA_GRAD = 'DADAPT_ADA_GRAD'
    DADAPT_ADAM = 'DADAPT_ADAM'
    DADAPT_ADAN = 'DADAPT_ADAN'
    DADAPT_LION = 'DADAPT_LION'
    DADAPT_SGD = 'DADAPT_SGD'

    # Prodigy
    PRODIGY = 'PRODIGY'

    # ADAFACTOR
    ADAFACTOR = 'ADAFACTOR'

    def __str__(self):
        return self.value
