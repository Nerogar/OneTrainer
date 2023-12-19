import random

from modules.util.params.BaseParams import BaseParams


class ConceptImageParams(BaseParams):
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

    def __init__(self, args: dict):
        super(ConceptImageParams, self).__init__(args)

    @staticmethod
    def default_values():
        args = {}

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

        return ConceptImageParams(args)


class ConceptTextParams(BaseParams):
    prompt_source: str
    prompt_path: str
    enable_tag_shuffling: bool
    tag_delimiter: str
    keep_tags_count: int

    def __init__(self, args: dict):
        super(ConceptTextParams, self).__init__(args)

    @staticmethod
    def default_values():
        args = {}

        args["prompt_source"] = "sample"
        args["prompt_path"] = ""
        args["enable_tag_shuffling"] = False
        args["tag_delimiter"] = ","
        args["keep_tags_count"] = 1

        return ConceptTextParams(args)


class ConceptParams(BaseParams):
    name: str
    path: str
    seed: int
    enabled: bool
    include_subdirectories: bool
    image_variations: int
    text_variations: int
    repeats: float

    image: ConceptImageParams
    text: ConceptTextParams

    def __init__(self, args: dict):
        super(ConceptParams, self).__init__(args)

    def to_dict(self):
        as_dict = super(ConceptParams, self).to_dict()
        as_dict['image'] = self.image.to_dict()
        as_dict['text'] = self.text.to_dict()
        return as_dict


    def from_dict(self, data):
        # detect old structure and translate
        if 'seed' not in data:
            translated_data = self.default_values()

            # @formatter:off
            translated_data.name = data['name'] if 'name' in data else translated_data.name
            translated_data.path = data['path'] if 'path' in data else translated_data.path
            translated_data.include_subdirectories = data['include_subdirectories'] if 'include_subdirectories' in data else translated_data.include_subdirectories

            translated_data.image.enable_crop_jitter = data['enable_crop_jitter'] if 'enable_crop_jitter' in data else translated_data.image.enable_crop_jitter
            translated_data.image.enable_random_flip = data['enable_random_flip'] if 'enable_random_flip' in data else translated_data.image.enable_random_flip
            translated_data.image.enable_random_rotate = data['enable_random_rotate'] if 'enable_random_rotate' in data else translated_data.image.enable_random_rotate
            translated_data.image.random_rotate_max_angle = data['random_rotate_max_angle'] if 'random_rotate_max_angle' in data else translated_data.image.random_rotate_max_angle
            translated_data.image.enable_random_brightness = data['enable_random_brightness'] if 'enable_random_brightness' in data else translated_data.image.enable_random_brightness
            translated_data.image.random_brightness_max_strength = data['random_brightness_max_strength'] if 'random_brightness_max_strength' in data else translated_data.image.random_brightness_max_strength
            translated_data.image.enable_random_contrast = data['enable_random_contrast'] if 'enable_random_contrast' in data else translated_data.image.enable_random_contrast
            translated_data.image.random_contrast_max_strength = data['random_contrast_max_strength'] if 'random_contrast_max_strength' in data else translated_data.image.random_contrast_max_strength
            translated_data.image.enable_random_saturation = data['enable_random_saturation'] if 'enable_random_saturation' in data else translated_data.image.enable_random_saturation
            translated_data.image.random_saturation_max_strength = data['random_saturation_max_strength'] if 'random_saturation_max_strength' in data else translated_data.image.random_saturation_max_strength
            translated_data.image.enable_random_hue = data['enable_random_hue'] if 'enable_random_hue' in data else translated_data.image.enable_random_hue
            translated_data.image.random_hue_max_strength = data['random_hue_max_strength'] if 'random_hue_max_strength' in data else translated_data.image.random_hue_max_strength

            translated_data.text.prompt_source = data['prompt_source'] if 'prompt_source' in data else translated_data.text.prompt_source
            translated_data.text.prompt_path = data['prompt_path'] if 'prompt_path' in data else translated_data.text.prompt_path
            translated_data.text.enable_tag_shuffling = data['enable_tag_shuffling'] if 'enable_tag_shuffling' in data else translated_data.text.enable_tag_shuffling
            translated_data.text.tag_delimiter = data['tag_delimiter'] if 'tag_delimiter' in data else translated_data.text.tag_delimiter
            translated_data.text.keep_tags_count = data['keep_tags_count'] if 'keep_tags_count' in data else translated_data.text.keep_tags_count
            # @formatter:on

            return translated_data
        else:
            concept_params = super(ConceptParams, self).from_dict(data)
            concept_params.image = ConceptImageParams(data['image'])
            concept_params.text = ConceptTextParams(data['text'])
            return concept_params

    @staticmethod
    def default_values():
        args = {}

        args["name"] = ""
        args["path"] = ""
        args["seed"] = random.randint(-(1 << 30), 1 << 30)
        args["enabled"] = True
        args["image_variations"] = 1
        args["text_variations"] = 1
        args["repeats"] = 1.0
        args["include_subdirectories"] = False

        concept_params = ConceptParams(args)
        concept_params.image = ConceptImageParams.default_values()
        concept_params.text = ConceptTextParams.default_values()

        return concept_params
