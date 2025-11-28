from modules.util.enum.BaseEnum import BaseEnum


class NoiseScheduler(BaseEnum):
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

    def pretty_print(self):
        return {
            NoiseScheduler.DDIM: 'DDIM',
            NoiseScheduler.EULER: 'Euler',
            NoiseScheduler.EULER_A: 'Euler A',
            NoiseScheduler.DPMPP: 'DPM++',
            NoiseScheduler.DPMPP_SDE: 'DPM++ SDE',
            NoiseScheduler.UNIPC: 'UniPC',
            NoiseScheduler.EULER_KARRAS: 'Euler Karras',
            NoiseScheduler.DPMPP_KARRAS: 'DPM++ Karras',
            NoiseScheduler.DPMPP_SDE_KARRAS: 'DPM++ SDE Karras',
            NoiseScheduler.UNIPC_KARRAS: 'UniPC Karras',
        }[self]

    @staticmethod
    def is_enabled(value, context=None):
        return value in [
            NoiseScheduler.DDIM,
            NoiseScheduler.EULER,
            NoiseScheduler.EULER_A,
            # NoiseScheduler.DPMPP, # TODO: produces noisy samples
            # NoiseScheduler.DPMPP_SDE, # TODO: produces noisy samples
            NoiseScheduler.UNIPC,
            NoiseScheduler.EULER_KARRAS,
            NoiseScheduler.DPMPP_KARRAS,
            NoiseScheduler.DPMPP_SDE_KARRAS,
            # NoiseScheduler.UNIPC_KARRAS,  # TODO: update diffusers to fix UNIPC_KARRAS (see https://github.com/huggingface/diffusers/pull/4581)
        ]
