from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.params.BaseParams import BaseParams


class SampleParams(BaseParams):
    enabled: bool
    prompt: str
    negative_prompt: str
    height: int
    width: int
    seed: int
    random_seed: bool
    diffusion_steps: int
    cfg_scale: float
    noise_scheduler: NoiseScheduler

    def __init__(self, args: dict):
        super(SampleParams, self).__init__(args)

    @staticmethod
    def default_values():
        args = {}

        args["enabled"] = True
        args["prompt"] = ""
        args["negative_prompt"] = ""
        args["height"] = 512
        args["width"] = 512
        args["seed"] = 42
        args["random_seed"] = False
        args["diffusion_steps"] = 20
        args["cfg_scale"] = 7.0
        args["noise_scheduler"] = NoiseScheduler.DDIM

        return SampleParams(args)
