import csv
import logging

from modules.module.captioning.models import BaseImageCaptionModel
from modules.module.captioning.sample import CaptionSample

import torch

import huggingface_hub
import numpy as np
import onnxruntime
from PIL import Image

logger = logging.getLogger(__name__)

class BooruModels(BaseImageCaptionModel):
    # Map model names to their HuggingFace repositories
    MODEL_REPOS = {
        "WD14 VIT v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
        "WD EVA02-Large Tagger v3": "SmilingWolf/wd-eva02-large-tagger-v3",
        "WD SwinV2 Tagger v3": "SmilingWolf/wd-swinv2-tagger-v3",
        "JoyTag": "fancyfeast/joytag",
    }

    # Default thresholds
    DEFAULT_GENERAL_THRESHOLD = 0.35
    DEFAULT_CHARACTER_THRESHOLD = 0.85
    DEFAULT_MIN_GENERAL_THRESHOLD = 0.35
    DEFAULT_MIN_CHARACTER_THRESHOLD = 0.15
    DEFAULT_JOYTAG_THRESHOLD = 0.387

    # Model-specific thresholds
    MODEL_THRESHOLDS = {
        "WD EVA02-Large Tagger v3": {"general": 0.5}, #EVA02 is highly overconfident/hallucinates
    }


    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_name: str | None = None,
        use_mcut_general: bool = False,
        use_mcut_character: bool = False,
        min_general_threshold: float = None,
        min_character_threshold: float = None,
        general_threshold: float = None,
        character_threshold: float = None,
        joytag_threshold: float = None,
    ):
        self.device = device
        self.dtype = dtype
        self.use_mcut_general = use_mcut_general
        self.use_mcut_character = use_mcut_character

        self.min_general_threshold = min_general_threshold if min_general_threshold is not None else self.DEFAULT_MIN_GENERAL_THRESHOLD
        self.min_character_threshold = min_character_threshold if min_character_threshold is not None else self.DEFAULT_MIN_CHARACTER_THRESHOLD
        self.general_threshold = general_threshold if general_threshold is not None else self.DEFAULT_GENERAL_THRESHOLD
        self.character_threshold = character_threshold if character_threshold is not None else self.DEFAULT_CHARACTER_THRESHOLD
        self.joytag_threshold = joytag_threshold if joytag_threshold is not None else self.DEFAULT_JOYTAG_THRESHOLD


        # For backward compatibility - default to WD14 VIT v2
        if model_name is None or model_name not in self.MODEL_REPOS:
            self.model_name = "WD14 VIT v2"
            logger.warning(f"WARNING: Unknown model name '{model_name}', defaulting to WD14 VIT v2")
        else:
            self.model_name = model_name

        # Apply model-specific threshold overrides if they exist
        if self.model_name in self.MODEL_THRESHOLDS:
            model_settings = self.MODEL_THRESHOLDS[self.model_name]
            if "general" in model_settings:
                self.general_threshold = model_settings["general"]

        logger.debug(f"Initializing BooruModels with model_name='{self.model_name}'")
        repo_id = self.MODEL_REPOS[self.model_name]
        logger.info(f"Loading {self.model_name} model from {repo_id}")

        # Wrap model download in try/except block
        try:
            model_path = huggingface_hub.hf_hub_download(repo_id, "model.onnx")
        except Exception as e:
            logger.error(f"Error downloading model.onnx from {repo_id}: {e}")
            raise
        if device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        try:
            self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        except Exception as e:
            logger.error(f"Error creating ONNX runtime session for model {self.model_name}: {e}")
            raise

        self.input = self.model.get_inputs()[0]
        self.output = self.model.get_outputs()[0]

        # Cache input dimensions for faster access
        self.input_shape = self.input.shape

        # Add better validation for input shape
        if len(self.input_shape) == 4:
            _, _, self.input_height, self.input_width = self.input_shape
            logger.debug(f"Model expects input shape: {self.input_shape}")
        else:
            # Default fallback dimensions
            logger.warning(f"Unexpected input shape: {self.input_shape}, using default dimensions")
            self.input_height = self.input_width = 448

        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []

        # JoyTag uses top_tags.txt instead of selected_tags.csv
        if self.model_name == "JoyTag":
            try:
                label_path = huggingface_hub.hf_hub_download(repo_id, "top_tags.txt")
            except Exception as e:
                logger.error(f"Error downloading top_tags.txt from {repo_id}: {e}")
                raise
            with open(label_path, "r", encoding="utf-8") as file:
                self.tag_names = [line.strip() for line in file.readlines()]
                # JoyTag doesn't have categories, all tags are considered general
                self.general_indexes = list(range(len(self.tag_names)))
                logger.debug(f"Loaded {len(self.tag_names)} JoyTag tags")
        else:
            # WD models use selected_tags.csv with categories
            try:
                label_path = huggingface_hub.hf_hub_download(repo_id, "selected_tags.csv")
            except Exception as e:
                logger.error(f"Error downloading selected_tags.csv from {repo_id}: {e}")
                raise
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
                logger.debug(f"Loaded {len(self.tag_names)} WD tags")

    def mcut_threshold(self, probs: np.ndarray) -> float:
        """
        Maximum Cut Thresholding (MCut) algorithm for determining optimal threshold value.

        Reference:
        Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
        for Multi-label Classification. In 11th International Symposium, IDA 2012
        (pp. 172-183).

        Args:
            probs: 1D numpy array of probabilities between 0 and 1, typically
                  representing confidence scores for a set of tags

        Returns:
            float: The calculated threshold value that maximizes the difference between
                   adjacent probability values. Returns 0.0 if the input array has 1 or fewer elements.
        """
        if len(probs) <= 1:
            return 0.0

        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return thresh

    def calculate_threshold(self, probs: np.ndarray, indexes: list[int], use_mcut: bool, base_threshold: float, min_threshold: float) -> float:
        """
        Calculate threshold based on MCut or use base threshold.

        Args:
            probs: 1D numpy array containing probabilities for all tags
            indexes: List of integer indices identifying which tags to consider from the probs array
            use_mcut: Boolean flag indicating whether to use MCut thresholding (True) or base threshold (False)
            base_threshold: Float value to use as threshold if not using MCut
            min_threshold: Minimum allowed threshold value when using MCut algorithm

        Returns:
            float: The calculated threshold value. If use_mcut=True, returns max(mcut_threshold, min_threshold).
                   If use_mcut=False or if tag_probs is empty, returns base_threshold.
        """
        if use_mcut:
            tag_probs = probs[indexes]
            if tag_probs.size > 0:
                mcut_thresh = self.mcut_threshold(tag_probs)
                threshold = max(mcut_thresh, min_threshold)
                logger.debug(f"Using MCut threshold: {threshold:.3f}")
                return threshold
        return base_threshold

    def process_tags(self, probs: np.ndarray, indexes: list[int], threshold: float) -> list[tuple[str, float]]:
        """
        Process tags based on threshold and return sorted labels.

        Args:
            probs: 1D numpy array containing probabilities for all tags
            indexes: List of integer indices identifying which tags to process from the probs array
            threshold: Float value representing the minimum probability threshold for including a tag

        Returns:
            list[tuple[str, float]]: A list of tuples containing (tag_name, probability),
                                    sorted in descending order of probability. Only includes
                                    tags with probability > threshold.
        """
        tags = np.array(self.tag_names)[indexes]
        tag_probs = probs[indexes]
        valid_mask = tag_probs > threshold
        valid_tags = tags[valid_mask]
        valid_probs = tag_probs[valid_mask]
        sort_indices = np.argsort(-valid_probs)
        sorted_tags = valid_tags[sort_indices]
        sorted_probs = valid_probs[sort_indices]
        return list(zip(sorted_tags, sorted_probs, strict=False))

    def format_tags(self, labels: list[tuple[str, float]]) -> list[str]:
        """
        Format tag labels and replace underscores with spaces.

        Args:
            labels: List of tuples with (tag_name, probability)

        Returns:
            List of formatted tag strings with underscores replaced by spaces
        """
        return [label[0].replace("_", " ") for label in labels]

    def preprocess_image_base(self, image: Image.Image, target_size: int, square_pad: bool = True) -> Image.Image:
        """
        Common image preprocessing: square padding and resizing

        Args:
            image: Input PIL image
            target_size: Target size for width and height
            square_pad: Whether to pad the image to square before resizing

        Returns:
            Preprocessed PIL image
        """
        if square_pad:
            # Pad image to square
            width, height = image.size
            max_dim = max(width, height)
            pad_left = (max_dim - width) // 2
            pad_top = (max_dim - height) // 2

            padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
            padded_image.paste(image, (pad_left, pad_top))
            image = padded_image

        # Resize to target size
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.BICUBIC)

        return image

    def preprocess_image_joytag(self, image: Image.Image) -> np.ndarray:
        """
        Special preprocessing for JoyTag following their official implementation

        Args:
            image: Input PIL image

        Returns:
            4D numpy array with shape [1, channels, height, width]
        """
        # Use common preprocessing for padding and resizing
        processed_image = self.preprocess_image_base(image, self.input_height)

        # JT specific normalization
        img_array = np.asarray(processed_image, dtype=np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_array = (img_array - mean) / std

        return img_array.transpose(2, 0, 1)[np.newaxis, :]

    def preprocess_image_wd(self, image: Image.Image) -> np.ndarray:
        """
        Preprocessing for WD models

        Args:
            image: Input PIL image

        Returns:
            4D numpy array with shape [1, height, width, channels] in BGR format
        """
        # WD models just need resize without padding
        processed_image = self.preprocess_image_base(
            image,
            target_size=self.input_height,
            square_pad=False
        )

        # Convert to BGR numpy array
        image_np = np.asarray(processed_image)
        image_np = image_np[:, :, ::-1]  # Convert from RGB to BGR
        image_np = image_np.astype(np.float32)
        image_np = np.expand_dims(image_np, 0)

        return image_np

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        initial: str = "",
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = "",
    ) -> str:
        """
        Generate a caption for the given image using the model

        Args:
            caption_sample: Sample containing the image to caption
            initial: Initial text to prepend (takes precedence over initial_caption)
            initial_caption: Alternative initial text if initial is empty
            caption_prefix: Text to add at the very beginning of the caption
            caption_postfix: Text to add at the very end of the caption

        Returns:
            Generated caption string with tags for the image

        Raises:
            ValueError: If model returns unexpected output shape
        """
        logger.debug(f"Generating caption with model: {self.model_name}")
        image = caption_sample.get_image()

        if self.model_name == "JoyTag":
            input_tensor = self.preprocess_image_joytag(image)
            input_name = self.input.name
            label_name = self.output.name

            # Run model and validate output
            raw_output = self.model.run([label_name], {input_name: input_tensor})[0]

            if not isinstance(raw_output, np.ndarray) or raw_output.ndim < 1:
                logger.error(f"Unexpected output from JoyTag model: {type(raw_output)}")
                raise ValueError("Model returned invalid output type: {type(raw_output)}")

            probs = 1.0 / (1.0 + np.exp(-raw_output))

            # Validate probability array shape
            if probs.ndim == 0 or probs.shape[0] == 0:
                logger.error(f"Empty probability array from JoyTag model: {probs.shape}")
                raise ValueError("Model returned empty probability array")

        else:  # WD models
            input_tensor = self.preprocess_image_wd(image)
            input_name = self.input.name
            label_name = self.output.name

            # Run model and validate output
            output = self.model.run([label_name], {input_name: input_tensor})

            if not output or not isinstance(output[0], np.ndarray):
                logger.error(f"Unexpected output from WD model: {type(output)}")
                raise ValueError(f"Model returned invalid output type: {type(output)}")

            probs = output[0]

            # Handle different output shapes gracefully
            if probs.ndim > 1:
                if probs.shape[0] > 0:
                    probs = probs[0].astype(float)
                else:
                    logger.error(f"Empty probability array from WD model: {probs.shape}")
                    raise ValueError("Model returned empty probability array")
            else:
                # If output is already 1D, use it directly
                probs = probs.astype(float)

        # Generate caption based on model type
        if self.model_name == "JoyTag":
            sorted_labels = self.process_tags(probs[0], self.general_indexes, self.joytag_threshold)
            all_tags = self.format_tags(sorted_labels)
            generated_caption = ", ".join(all_tags)
        else:
            # Make sure we have enough probabilities
            if len(probs) < max(max(self.rating_indexes, default=-1),
                              max(self.general_indexes, default=-1),
                              max(self.character_indexes, default=-1)) + 1:
                logger.error(f"Insufficient probabilities ({len(probs)}) for tag indices")
                raise ValueError("Model returned insufficient probabilities for tag processing")

            labels = list(zip(self.tag_names, probs, strict=False))
            ratings_names = [labels[i] for i in self.rating_indexes]
            ratings_dict = dict(ratings_names)
            top_rating = max(ratings_dict.items(), key=lambda x: x[1]) if ratings_dict else ("unknown", 0.0)
            rating_tag = top_rating[0].replace("_", " ")

            current_general_threshold = self.general_threshold
            current_character_threshold = self.character_threshold

            general_threshold = self.calculate_threshold(
                probs, self.general_indexes,
                self.use_mcut_general,
                current_general_threshold,
                self.min_general_threshold
            )
            character_threshold = self.calculate_threshold(
                probs, self.character_indexes,
                self.use_mcut_character,
                current_character_threshold,
                self.min_character_threshold
            )

            sorted_general_labels = self.process_tags(probs, self.general_indexes, general_threshold)
            sorted_character_labels = self.process_tags(probs, self.character_indexes, character_threshold)
            all_tags = [rating_tag]
            all_tags += self.format_tags(sorted_character_labels)
            all_tags += self.format_tags(sorted_general_labels)
            generated_caption = ", ".join(all_tags)

        # Add the initial caption; prioritize `initial` if present
        if initial:
            generated_caption = f"{initial} {generated_caption}"
        elif initial_caption:
            generated_caption = f"{initial_caption} {generated_caption}"

        # Add prefix and postfix
        final_caption = (caption_prefix + generated_caption + caption_postfix).strip()
        return final_caption
