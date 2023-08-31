from modules.util.params.BaseParams import BaseParams


class SampleParams(BaseParams):
    prompt: str
    negative_prompt: str
    height: int
    width: int
    seed: int
    diffusion_steps: int
    cfg_scale: float

    def __init__(self, args: dict):
        super(SampleParams, self).__init__(args)

    @staticmethod
    def default_values():
        args = {}

        args["prompt"] = ""
        args["negative_prompt"] = ""
        args["height"] = 512
        args["width"] = 512
        args["seed"] = 42
        args["diffusion_steps"] = 20
        args["cfg_scale"] = 7.0

        return SampleParams(args)
