from enum import Enum


class Optimizer(Enum):
    SGD = 'SGD'
    ADAM = 'ADAM'
    ADAMW = 'ADAMW'
    ADAM_8BIT = 'ADAM_8BIT'
    ADAMW_8BIT = 'ADAMW_8BIT'
    Lion = 'Lion'
    DAdaptSGD = "DAdaptSGD"
    DAdaptAdam = "DAdaptAdam"
    DAdaptAdan = "DAdaptAdan"
    DAdaptAd = "DAdaptAdan"
    DAdaptAdaGrad = "DAdaptAdaGrad"
    DAdaptLion = "DAdaptLion"

    def __str__(self):
        return self.value
