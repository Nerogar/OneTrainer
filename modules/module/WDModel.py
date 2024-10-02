import csv

from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample

import torch

import huggingface_hub
import numpy as np
import onnxruntime


class WDModel(BaseImageCaptionModel):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        model_path = huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "model.onnx"
        )
        if device.type == 'cpu':
            provider = "CPUExecutionProvider"
        else:
            provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "CPUExecutionProvider"
        self.model = onnxruntime.InferenceSession(model_path, providers=[provider])

        label_path = huggingface_hub.hf_hub_download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2", "selected_tags.csv"
        )
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []
        with open(label_path, newline='') as file:
            reader = csv.DictReader(file, delimiter=',', quotechar='\"')
            for i, row in enumerate(reader):
                if row["category"] == "9":
                    self.rating_indexes.append(i)
                if row["category"] == "0":
                    self.general_indexes.append(i)
                if row["category"] == "4":
                    self.character_indexes.append(i)

                self.tag_names.append(row["name"])

    def generate_caption(
            self,
            caption_sample: CaptionSample,
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
    ):
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

        sorted_general_labels = sorted(general_labels, key=lambda label: label[1], reverse=True)
        predicted_caption = ", ".join([
            label[0].replace("_", " ")
            for label
            in sorted_general_labels
        ])
        predicted_caption = (caption_prefix + predicted_caption + caption_postfix).strip()

        return predicted_caption
