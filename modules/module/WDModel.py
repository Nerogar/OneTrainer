import csv

import huggingface_hub
import numpy as np
import onnxruntime
import torch

from modules.module.BaseImageCaptionModel import CaptionSample, BaseImageCaptionModel


class WDModel(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        model_path = huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "model.onnx"
        )
        self.model = onnxruntime.InferenceSession(model_path)

        labeL_path = huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "selected_tags.csv"
        )
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []
        with open(labeL_path, newline='') as file:
            reader = csv.DictReader(file, delimiter=',', quotechar='\"')
            for i, row in enumerate(reader):
                if row["category"] == "9":
                    self.rating_indexes.append(i)
                if row["category"] == "0":
                    self.general_indexes.append(i)
                if row["category"] == "4":
                    self.character_indexes.append(i)

                self.tag_names.append(row["name"])

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

        _, height, width, _ = self.model.get_inputs()[0].shape

        image = caption_sample.get_image()
        image = image.resize((width, height))
        image = np.asarray(image)
        image = image[:, :, ::-1]  # RGB to BGR
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0]
        probs = probs[0].astype(float)

        general_labels = [(self.tag_names[i], probs[i]) for i in self.general_indexes if probs[i] > 0.35]
        character_labels = [(self.tag_names[i], probs[i]) for i in self.character_indexes if probs[i] > 0.8]

        sorted_general_labels = sorted(general_labels, key=lambda label: label[1], reverse=True)
        predicted_caption = ", ".join([
            label[0].replace("_", " ")
            for label
            in sorted_general_labels
        ])

        if mode == 'replace' or mode == 'fill':
            caption_sample.set_caption(predicted_caption)

        caption_sample.save_caption()
