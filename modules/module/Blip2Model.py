import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from modules.module.BaseImageCaptionModel import CaptionSample, BaseImageCaptionModel


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

    def caption_image(
            self,
            filename: str,
            initial_caption: str = "",
            mode: str = 'fill',
    ):
        caption_sample = CaptionSample(filename)

        existing_caption = caption_sample.get_caption()
        if mode == 'fill' and existing_caption is not None and existing_caption != "":
            return

        inputs = self.processor(caption_sample.get_image(), initial_caption, return_tensors="pt")
        inputs = inputs.to(self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        predicted_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        predicted_caption = (initial_caption + predicted_caption).strip()

        if mode == 'replace' or mode == 'fill':
            caption_sample.set_caption(predicted_caption)

        caption_sample.save_caption()
