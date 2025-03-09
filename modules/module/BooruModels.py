import csv

from modules.module.BaseImageCaptionModel import (
    BaseImageCaptionModel,
    CaptionSample,
)

import torch

import huggingface_hub
import numpy as np
import onnxruntime
from PIL import Image


class BooruModels(BaseImageCaptionModel):
    # Map model names to their HuggingFace repositories
    MODEL_REPOS = {
        "WD14 VIT v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "WD EVA02-Large Tagger v3": "SmilingWolf/wd-eva02-large-tagger-v3",
        "WD SwinV2 Tagger v3": "SmilingWolf/wd-swinv2-tagger-v3",
        "JoyTag": "fancyfeast/joytag",
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
            print(f"WARNING: Unknown model name '{model_name}', defaulting to WD14 VIT v2")
        else:
            self.model_name = model_name

        print(f"Initializing BooruModels with model_name='{self.model_name}'")
        repo_id = self.MODEL_REPOS[self.model_name]
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

        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []

        # JoyTag uses top_tags.txt instead of selected_tags.csv
        if self.model_name == "JoyTag":
            label_path = huggingface_hub.hf_hub_download(repo_id, "top_tags.txt")
            with open(label_path, "r", encoding="utf-8") as file:
                self.tag_names = [line.strip() for line in file.readlines()]
                # JoyTag doesn't have categories, all tags are considered general
                self.general_indexes = list(range(len(self.tag_names)))
                print(f"Loaded {len(self.tag_names)} JoyTag tags")
        else:

            # WD models use selected_tags.csv with categories
            label_path = huggingface_hub.hf_hub_download(repo_id, "selected_tags.csv")
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
                print(f"Loaded {len(self.tag_names)} WD tags")

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

    def calculate_threshold(self, probs, indexes, use_mcut, base_threshold, min_threshold):
        """Calculate threshold based on MCut or use base threshold"""
        if use_mcut:
            tag_probs = np.array([probs[i] for i in indexes])
            if len(tag_probs) > 0:
                mcut_thresh = self.mcut_threshold(tag_probs)
                threshold = max(mcut_thresh, min_threshold)
                print(f"Using MCut threshold: {threshold:.3f}")
                return threshold
        return base_threshold

    def process_tags(self, probs, indexes, threshold):
        """Process tags based on threshold and return sorted labels"""
        labels = [
            (self.tag_names[i], probs[i])
            for i in indexes
            if probs[i] > threshold
        ]
        return sorted(labels, key=lambda label: label[1], reverse=True)

    def format_tags(self, labels):
        """Format tag labels and replace underscores with spaces"""
        return [label[0].replace("_", " ") for label in labels]

    def preprocess_image_joytag(self, image):
        """
        Special preprocessing for JoyTag following their official implementation
        """
        # Get model input shape
        input_shape = self.model.get_inputs()[0].shape
        if len(input_shape) == 4:
            _, _, target_height, target_width = input_shape
            target_size = target_height
        else:
            # Fallback to 448 as per JoyTag's official implementation
            target_size = 448

        # Pad image to square
        width, height = image.size
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2

        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize to target size
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to numpy array
        img_array = np.asarray(padded_image, dtype=np.float32) / 255.0

        # Apply JoyTag's normalization values
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        img_array = (img_array - mean) / std

        # Add batch dimension and reorder to NCHW for ONNX
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)

        return img_array

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = "",
    ):
        """Generate a caption for the given image using the model"""
        print(f"Generating caption with model: {self.model_name}")

        image = caption_sample.get_image()

        # Different preprocessing for JoyTag vs WD models
        if self.model_name == "JoyTag":
            # JoyTag-specific preprocessing
            input_tensor = self.preprocess_image_joytag(image)
            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name

            # Run inference
            raw_output = self.model.run([label_name], {input_name: input_tensor})[0]

            # Apply sigmoid to convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-raw_output))
        else:
            # Standard WD model preprocessing
            _, height, width, _ = self.model.get_inputs()[0].shape
            image = image.resize((width, height))
            image = np.asarray(image)
            image = image[:, :, ::-1]  # RGB to BGR
            image = image.astype(np.float32)
            image = np.expand_dims(image, 0)

            input_name = self.model.get_inputs()[0].name
            label_name = self.model.get_outputs()[0].name
            probs = self.model.run([label_name], {input_name: image})[0]
            probs = probs[0].astype(float)

        # Process predictions based on model type
        if self.model_name == "JoyTag":
            # JoyTag uses a fixed threshold of 0.4 according to official implementation,
            # after benchmarking it I found for Onnx model 0.387 to be equivalent
            threshold = 0.387
            sorted_labels = self.process_tags(probs[0], self.general_indexes, threshold)  # Use probs[0] here
            all_tags = self.format_tags(sorted_labels)
            generated_caption = ", ".join(all_tags)
        else:
            # WD models with category-specific processing
            labels = list(zip(self.tag_names, probs, strict=False))

            # First handle ratings: pick one with argmax
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

            # Calculate thresholds using helper function
            general_threshold = self.calculate_threshold(
                probs, self.general_indexes,
                self.use_mcut_general,
                general_threshold,
                self.min_general_threshold
            )

            character_threshold = self.calculate_threshold(
                probs, self.character_indexes,
                self.use_mcut_character,
                character_threshold,
                self.min_character_threshold
            )

            # Process tags using helper functions
            sorted_general_labels = self.process_tags(probs, self.general_indexes, general_threshold)
            sorted_character_labels = self.process_tags(probs, self.character_indexes, character_threshold)

            # Format tags and combine
            all_tags = [rating_tag]
            all_tags += self.format_tags(sorted_character_labels)
            all_tags += self.format_tags(sorted_general_labels)

            # Join all tags
            generated_caption = ", ".join(all_tags)

        # Add initial caption if provided
        if initial_caption:
            generated_caption = f"{initial_caption} {generated_caption}"

        # Add prefix and postfix
        final_caption = (
            caption_prefix + generated_caption + caption_postfix
        ).strip()

        return final_caption
