import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from modules.module.BaseImageCaptionModel import CaptionSample, BaseImageCaptionModel


class BlipModel(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
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

        if mode == 'replace' or mode == 'fill':
            caption_sample.set_caption(predicted_caption)

        caption_sample.save_caption()
