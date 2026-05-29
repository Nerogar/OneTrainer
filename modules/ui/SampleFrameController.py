from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.ModelType import ModelType


class SampleFrameController:
    def __init__(self, sample: SampleConfig, model_type: ModelType):
        self.sample = sample
        self.model_type = model_type

    def is_flow_matching(self) -> bool:
        return self.model_type.is_flow_matching()

    def is_inpainting_model(self) -> bool:
        return self.model_type.has_conditioning_image_input()

    def is_video_model(self) -> bool:
        return self.model_type.is_video_model()
