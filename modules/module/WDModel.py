import csv

from modules.module.BaseImageCaptionModel import (
    BaseImageCaptionModel,
    CaptionSample,
)

import torch

import huggingface_hub
import numpy as np
import onnxruntime


class WDModel(BaseImageCaptionModel):
    # Map model names to their HuggingFace repositories
    MODEL_REPOS = {
        "WD14 VIT v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "WD EVA02-Large Tagger v3": "SmilingWolf/wd-eva02-large-tagger-v3",
        "WD SwinV2 Tagger v3": "SmilingWolf/wd-swinv2-tagger-v3",
    }

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_name: str | None = None,
        use_mcut_general: bool = False,
        use_mcut_character: bool = False,
        min_general_threshold: float = 0.35,
        min_character_threshold: float = 0.15,
    ):
        self.device = device
        self.dtype = dtype
        self.use_mcut_general = use_mcut_general
        self.use_mcut_character = use_mcut_character
        self.min_general_threshold = min_general_threshold
        self.min_character_threshold = min_character_threshold

        # For backward compatibility - default to WD14 VIT v2
        if model_name is None or model_name not in self.MODEL_REPOS:
            self.model_name = "WD14 VIT v2"
        else:
            self.model_name = model_name

        repo_id = self.MODEL_REPOS[self.model_name]

        # Log which model is being loaded
        print(f"Loading {self.model_name} model from {repo_id}")

        model_path = huggingface_hub.hf_hub_download(repo_id, "model.onnx")
        if device.type == "cpu":
            provider = "CPUExecutionProvider"
        else:
            provider = (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider"
                in onnxruntime.get_available_providers()
                else "CPUExecutionProvider"
            )
        self.model = onnxruntime.InferenceSession(
            model_path, providers=[provider]
        )

        label_path = huggingface_hub.hf_hub_download(
            repo_id, "selected_tags.csv"
        )
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []
        with open(label_path, newline="") as file:
            reader = csv.DictReader(file, delimiter=",", quotechar='"')
            for i, row in enumerate(reader):
                if row["category"] == "9":
                    self.rating_indexes.append(i)
                if row["category"] == "0":
                    self.general_indexes.append(i)
                if row["category"] == "4":
                    self.character_indexes.append(i)

                self.tag_names.append(row["name"])

    def mcut_threshold(self, probs):
        """
        Maximum Cut Thresholding (MCut)
        Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
         for Multi-label Classification. In 11th International Symposium, IDA 2012
         (pp. 172-183).
        """
        if len(probs) <= 1:
            return 0.0

        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return thresh

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = "",
    ):
        """Generate a caption for the given image using the WD model"""
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

        # Process all predictions organized by tag type
        labels = list(zip(self.tag_names, probs, strict=False))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        ratings_dict = dict(ratings_names)
        # Get the top rating with highest confidence
        top_rating = max(ratings_dict.items(), key=lambda x: x[1])
        rating_tag = top_rating[0].replace("_", " ")

        # Set base thresholds
        general_threshold = 0.35  # Default for WD14 VIT v2 and SwinV2
        character_threshold = 0.85

        # Only EVA02 needs a different general threshold
        if "EVA02" in self.model_name:
            general_threshold = 0.5

        # Apply MCut thresholding if requested
        if self.use_mcut_general:
            general_probs = np.array(
                [probs[i] for i in self.general_indexes]
            )
            if len(general_probs) > 0:
                mcut_thresh = self.mcut_threshold(general_probs)
                general_threshold = max(
                    mcut_thresh, self.min_general_threshold
                )
                print(
                    f"Using MCut general threshold: {general_threshold:.3f}"
                )

        if self.use_mcut_character:
            character_probs = np.array(
                [probs[i] for i in self.character_indexes]
            )
            if len(character_probs) > 0:
                mcut_thresh = self.mcut_threshold(character_probs)
                # Always enforce a minimum threshold of 0.15 for characters when using MCut
                character_threshold = max(
                    mcut_thresh, 0.15, self.min_character_threshold
                )
                print(
                    f"Using MCut character threshold: {character_threshold:.3f}"
                )

        # Get general tags above threshold
        general_labels = [
            (self.tag_names[i], probs[i])
            for i in self.general_indexes
            if probs[i] > general_threshold
        ]

        # Get character tags above threshold
        character_labels = [
            (self.tag_names[i], probs[i])
            for i in self.character_indexes
            if probs[i] > character_threshold
        ]

        # Sort general tags by confidence
        sorted_general_labels = sorted(
            general_labels, key=lambda label: label[1], reverse=True
        )

        # Sort character tags by confidence
        sorted_character_labels = sorted(
            character_labels, key=lambda label: label[1], reverse=True
        )

        # Start with rating tag, followed by character tags and general tags
        all_tags = [rating_tag]
        all_tags += [
            label[0].replace("_", " ") for label in sorted_character_labels
        ]
        all_tags += [
            label[0].replace("_", " ") for label in sorted_general_labels
        ]

        # Join all tags, notably without escaping parentheses like the author does
        # because we are using this for training not prompting.
        generated_caption = ", ".join(all_tags)

        # Add initial caption if provided
        if initial_caption:
            generated_caption = f"{initial_caption} {generated_caption}"

        # Add prefix and postfix
        final_caption = (
            caption_prefix + generated_caption + caption_postfix
        ).strip()

        return final_caption
