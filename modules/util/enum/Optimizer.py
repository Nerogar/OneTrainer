from enum import Enum


class Optimizer(Enum):
    SGD = 'SGD'
    ADAM = 'ADAM'
    ADAMW = 'ADAMW'
    ADAM_8BIT = 'ADAM_8BIT'
    ADAMW_8BIT = 'ADAMW_8BIT'
    LION = 'LION'
    DADAPT_SGD = 'DADAPT_SGD'
    DADAPT_ADAM = 'DADAPT_ADAM'
    DADAPT_ADAN = 'DADAPT_ADAN'
    DADAPT_ADA_GRAD = 'DADAPT_ADA_GRAD'
    DADAPT_LION = 'DADAPT_LION'

    def __str__(self):
        return self.value
