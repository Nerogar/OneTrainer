import os
from typing import Callable

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from modules.module.BaseImageCaptionModel import CaptionSample
from modules.util import path_util

DEVICE = "cuda"
DTYPE = torch.float16


class Blip2Model:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=DTYPE)
        self.model.eval()
        self.model.to(DEVICE)

    @staticmethod
    def __get_sample_filenames(sample_dir: str) -> [str]:
        filenames = []
        for filename in os.listdir(sample_dir):
            ext = os.path.splitext(filename)[1]
            if path_util.is_supported_image_extension(ext) and '-masklabel.png' not in filename:
                filenames.append(os.path.join(sample_dir, filename))

        return filenames

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
        inputs = inputs.to(DEVICE, DTYPE)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        predicted_caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        predicted_caption = (initial_caption + predicted_caption).strip()

        if mode == 'replace' or mode == 'fill':
            caption_sample.set_caption(predicted_caption)

        caption_sample.save_caption()

    def caption_images(
            self,
            filenames: [str],
            initial_caption: str = "",
            mode: str = 'fill',
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
    ):
        if progress_callback is not None:
            progress_callback(0, len(filenames))
        for i, filename in enumerate(tqdm(filenames)):
            try:
                self.caption_image(filename, initial_caption, mode)
            except Exception as e:
                if error_callback is not None:
                    error_callback(filename)
            if progress_callback is not None:
                progress_callback(i + 1, len(filenames))

    def caption_folder(
            self,
            sample_dir: str,
            initial_caption: str = "",
            mode: str = 'fill',
            progress_callback: Callable[[int, int], None] = None,
            error_callback: Callable[[str], None] = None,
    ):
        filenames = self.__get_sample_filenames(sample_dir)
        self.caption_images(
            filenames=filenames,
            initial_caption=initial_caption,
            mode=mode,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )
