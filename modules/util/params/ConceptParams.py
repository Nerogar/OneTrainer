from modules.util.params.BaseParams import BaseParams


class ConceptParams(BaseParams):
    name: str
    path: str
    prompt_source: str
    prompt_path: str
    enable_crop_jitter: bool
    enable_random_flip: bool
    enable_random_rotate: bool
    random_rotate_max_angle: float
    enable_random_brightness: bool
    random_brightness_max_strength: float
    enable_random_contrast: bool
    random_contrast_max_strength: float
    enable_random_saturation: bool
    random_saturation_max_strength: float
    enable_random_hue: bool
    random_hue_max_strength: float
    include_subdirectories: bool

    def __init__(self, args: dict):
        super(ConceptParams, self).__init__(args)

    @staticmethod
    def default_values():
        args = {}

        args["name"] = ""
        args["path"] = ""
        args["prompt_source"] = "sample"
        args["prompt_path"] = ""
        args["enable_crop_jitter"] = True
        args["enable_random_flip"] = True
        args["enable_random_rotate"] = False
        args["random_rotate_max_angle"] = 0.0
        args["enable_random_brightness"] = False
        args["random_brightness_max_strength"] = 0.0
        args["enable_random_contrast"] = False
        args["random_contrast_max_strength"] = 0.0
        args["enable_random_saturation"] = False
        args["random_saturation_max_strength"] = 0.0
        args["enable_random_hue"] = False
        args["random_hue_max_strength"] = 0.0
        args["include_subdirectories"] = False

        return ConceptParams(args)
