from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample

import torch

from transformers import AutoProcessor, Blip2ForConditionalGeneration


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
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
    ) -> str:
        inputs = self.processor(caption_sample.get_image(), initial_caption, return_tensors="pt")
        inputs = inputs.to(self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        predicted_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        predicted_caption = (caption_prefix + initial_caption + predicted_caption + caption_postfix).strip()

        return predicted_caption
