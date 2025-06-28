from modules.module.captioning.BaseImageCaptionModel import BaseImageCaptionModel
from modules.module.captioning.CaptionSample import CaptionSample

import torch

from transformers import AutoProcessor, Blip2ForConditionalGeneration

from .captioning.caption_config_types import BlipGenerationConfig


class Blip2Model(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=self.dtype
        )
        self.model.eval()
        self.model.to(self.device)

    def generate_caption(
            self,
            caption_sample: CaptionSample,
            prompt: str, # This was 'initial_caption'
            generation_config: BlipGenerationConfig | None = None # For interface consistency
    ) -> str:
        inputs = self.processor(caption_sample.get_image(), prompt, return_tensors="pt") # Use prompt
        inputs = inputs.to(self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        predicted_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Prefix/postfix are handled by BaseImageCaptionModel.caption_image
        return predicted_caption.strip()
