from enum import Enum


class NoiseScheduler(Enum):
    DDIM = 'DDIM'

    EULER = 'EULER'
    EULER_A = 'EULER_A'
    DPMPP = 'DPMPP'
    DPMPP_SDE = 'DPMPP_SDE'
    UNIPC = 'UNIPC'

    EULER_KARRAS = 'EULER_KARRAS'
    DPMPP_KARRAS = 'DPMPP_KARRAS'
    DPMPP_SDE_KARRAS = 'DPMPP_SDE_KARRAS'
    UNIPC_KARRAS = 'UNIPC_KARRAS'

    def __str__(self):
        return self.value
