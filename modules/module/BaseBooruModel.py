import csv
import logging

from modules.module.captioning.BaseImageCaptionModel import BaseImageCaptionModel
from modules.module.captioning.CaptionSample import CaptionSample

import torch

import huggingface_hub
import numpy as np
import onnxruntime
from PIL import Image

from .captioning.caption_config_types import WDGenerationConfig

logger = logging.getLogger(__name__)


class BaseBooruModel(BaseImageCaptionModel):
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
        "WD EVA02-Large Tagger v3": {
            "general": 0.5
        },  # EVA02 is highly overconfident/hallucinates
    }

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_name: str,  # Subclasses will provide this
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
        self.model_name = (
            model_name  # Set by subclass, used for MODEL_THRESHOLDS
        )
        self.use_mcut_general = use_mcut_general
        self.use_mcut_character = use_mcut_character

        # Initialize with defaults or user-provided values first
        self.min_general_threshold = (
            min_general_threshold
            if min_general_threshold is not None
            else self.DEFAULT_MIN_GENERAL_THRESHOLD
        )
        self.min_character_threshold = (
            min_character_threshold
            if min_character_threshold is not None
            else self.DEFAULT_MIN_CHARACTER_THRESHOLD
        )
        self.general_threshold = (
            general_threshold
            if general_threshold is not None
            else self.DEFAULT_GENERAL_THRESHOLD
        )
        self.character_threshold = (
            character_threshold
            if character_threshold is not None
            else self.DEFAULT_CHARACTER_THRESHOLD
        )
        self.joytag_threshold = (
            joytag_threshold
            if joytag_threshold is not None
            else self.DEFAULT_JOYTAG_THRESHOLD
        )

        # Apply model-specific threshold overrides if they exist (these take precedence)
        if self.model_name in self.MODEL_THRESHOLDS:
            model_settings = self.MODEL_THRESHOLDS[self.model_name]
            if "general" in model_settings:
                self.general_threshold = model_settings["general"]
            # Add similar checks if other thresholds (character, min_general, etc.) are added to MODEL_THRESHOLDS

        # These will be initialized by subclasses in _load_model_and_tags
        self.model = None
        self.input = None
        self.output = None
        self.input_height = 0
        self.input_width = 0
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []

        # Subclasses are responsible for calling _load_model_and_tags()
        # logger.debug(f"BaseBooruModel initialized for model_name='{self.model_name}'")

    def _load_model_and_tags(self):
        """
        Loads the ONNX model and associated tag files.
        Implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _load_model_and_tags"
        )

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
        # print(f"thresh type: {type(thresh)}, value: {thresh}") # Original print, kept for consistency
        return thresh

    def calculate_threshold(
        self,
        probs: np.ndarray,
        indexes: list[int],
        use_mcut: bool,
        base_threshold: float,
        min_threshold: float,
    ) -> float:
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

    def process_tags(
        self, probs: np.ndarray, indexes: list[int], threshold: float
    ) -> list[tuple[str, float]]:
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

    def preprocess_image_base(
        self, image: Image.Image, target_size: int, square_pad: bool = True
    ) -> Image.Image:
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

            padded_image = Image.new(
                "RGB", (max_dim, max_dim), (255, 255, 255)
            )
            padded_image.paste(image, (pad_left, pad_top))
            image = padded_image

        # Resize to target size
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.BICUBIC)

        return image

    def _preprocess_image_for_model(
        self, image: Image.Image
    ) -> np.ndarray:
        """
        Preprocesses the image for the specific model.
        Implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _preprocess_image_for_model"
        )

    def _run_model_and_get_probs(
        self, input_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Runs the model with the input tensor and returns probabilities.
        Implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _run_model_and_get_probs"
        )

    def _convert_probs_to_tags(self, probs: np.ndarray) -> list[str]:
        """
        Converts model output probabilities to a list of tag strings.
        Implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _convert_probs_to_tags"
        )

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        prompt: str,
        generation_config: WDGenerationConfig | None = None
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
            ValueError: If model returns unexpected output shape or insufficient probabilities
        """
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call _load_model_and_tags in subclass __init__."
            )

        logger.debug(f"Generating caption with model: {self.model_name}")
        image = caption_sample.get_image()

        input_tensor = self._preprocess_image_for_model(image)
        probs = self._run_model_and_get_probs(input_tensor)

        # Ensure probs is not empty before proceeding
        if probs is None or probs.size == 0:
            logger.error(
                f"Probability array from model {self.model_name} is empty or None."
            )
            raise ValueError(
                "Model returned empty or None probability array"
            )

        all_tags_list = self._convert_probs_to_tags(probs)
        generated_caption = ", ".join(all_tags_list)

        # Add the prompt if provided
        if prompt:
            generated_caption = f"{prompt} {generated_caption}"
        # Prefix and postfix are handled by BaseImageCaptionModel.caption_image
        return generated_caption.strip()


class WDBooruModel(BaseBooruModel):
    VALID_WD_MODELS = [
        "WD14 VIT v2",
        "WD EVA02-Large Tagger v3",
        "WD SwinV2 Tagger v3",
    ]

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_name: str = "WD14 VIT v2",
        use_mcut_general: bool = False,
        use_mcut_character: bool = False,
        min_general_threshold: float = None,
        min_character_threshold: float = None,
        general_threshold: float = None,
        character_threshold: float = None,
        # joytag_threshold is not used by WD models directly but passed to super for completeness
        joytag_threshold: float = None,
    ):
        if model_name not in self.VALID_WD_MODELS:
            logger.warning(
                f"Specified model_name '{model_name}' is not a recognized WD model. "
                f"Defaulting to 'WD14 VIT v2'. Valid models are: {self.VALID_WD_MODELS}"
            )
            model_name = "WD14 VIT v2"

        super().__init__(
            device,
            dtype,
            model_name,
            use_mcut_general,
            use_mcut_character,
            min_general_threshold,
            min_character_threshold,
            general_threshold,
            character_threshold,
            joytag_threshold,
        )
        logger.debug(
            f"Initializing WDBooruModel with model_name='{self.model_name}'"
        )
        self._load_model_and_tags()

    def _load_model_and_tags(self):
        repo_id = self.MODEL_REPOS[self.model_name]
        logger.info(f"Loading {self.model_name} model from {repo_id}")

        try:
            model_path = huggingface_hub.hf_hub_download(
                repo_id, "model.onnx"
            )
        except Exception as e:
            logger.error(
                f"Error downloading model.onnx from {repo_id}: {e}"
            )
            raise

        if self.device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider"
                in onnxruntime.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        try:
            self.model = onnxruntime.InferenceSession(
                model_path, providers=providers
            )
        except Exception as e:
            logger.error(
                f"Error creating ONNX runtime session for model {self.model_name}: {e}"
            )
            raise

        self.input = self.model.get_inputs()[0]
        self.output = self.model.get_outputs()[0]
        self.input_shape = self.input.shape

        if len(self.input_shape) == 4:
            # WD models typically have shape [batch_size, height, width, channels]
            # or [batch_size, channels, height, width] - check specific model
            # For WD SwinV2, it's [1, 384, 384, 3]. For WD ViT, it's [1, 448, 448, 3]
            # The original code used _, _, self.input_height, self.input_width assuming NCHW
            # but WD models are NHWC. Let's adapt.
            if (
                self.input_shape[1] == 3 or self.input_shape[3] == 3
            ):  # Check for channels dim
                # Assuming NHWC [batch, height, width, channels] for WD models like wd-v1-4-vit-tagger-v2
                if self.input_shape[3] == 3:  # NHWC
                    _, self.input_height, self.input_width, _ = (
                        self.input_shape
                    )
                # Assuming NCHW [batch, channels, height, width] for some other models if necessary
                elif self.input_shape[1] == 3:  # NCHW
                    _, _, self.input_height, self.input_width = (
                        self.input_shape
                    )
                else:  # Fallback if channel dim is not 3
                    logger.warning(
                        f"Unexpected channel dimension in input shape: {self.input_shape}, using H and W directly if possible."
                    )
                    self.input_height = self.input_shape[1]  # Or 2
                    self.input_width = self.input_shape[2]  # Or 3
            else:  # Default fallback dimensions
                logger.warning(
                    f"Unexpected input shape format: {self.input_shape}, using default dimensions 448x448"
                )
                self.input_height = self.input_width = 448
            logger.debug(
                f"Model expects input height: {self.input_height}, width: {self.input_width}"
            )

        else:
            logger.warning(
                f"Unexpected input shape rank: {self.input_shape}, using default dimensions 448x448"
            )
            self.input_height = self.input_width = 448

        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []
        try:
            label_path = huggingface_hub.hf_hub_download(
                repo_id, "selected_tags.csv"
            )
        except Exception as e:
            logger.error(
                f"Error downloading selected_tags.csv from {repo_id}: {e}"
            )
            raise
        with open(
            label_path, newline="", encoding="utf-8"
        ) as file:  # Added encoding
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

    def preprocess_image_wd(self, image: Image.Image) -> np.ndarray:
        """
        Preprocessing for WD models

        Args:
            image: Input PIL image

        Returns:
            4D numpy array with shape [1, height, width, channels] in BGR format
        """
        # WD models just need resize without padding
        # Use self.input_height which is determined during model loading
        processed_image = self.preprocess_image_base(
            image,
            target_size=self.input_height,  # Use loaded model's input size
            square_pad=False,
        )

        # Convert to BGR numpy array
        image_np = np.asarray(processed_image)
        image_np = image_np[:, :, ::-1]  # Convert from RGB to BGR
        image_np = image_np.astype(np.float32)
        image_np = np.expand_dims(image_np, 0)

        return image_np

    def _preprocess_image_for_model(
        self, image: Image.Image
    ) -> np.ndarray:
        return self.preprocess_image_wd(image)

    def _run_model_and_get_probs(
        self, input_tensor: np.ndarray
    ) -> np.ndarray:
        input_name = self.input.name
        label_name = self.output.name
        output = self.model.run([label_name], {input_name: input_tensor})

        if not output or not isinstance(output[0], np.ndarray):
            logger.error(
                f"Unexpected output from WD model: {type(output)}"
            )
            raise ValueError(
                f"Model returned invalid output type: {type(output)}"
            )

        probs = output[0]
        if probs.ndim > 1:
            if probs.shape[0] > 0:
                probs = probs[0].astype(float)
            else:
                logger.error(
                    f"Empty probability array from WD model: {probs.shape}"
                )
                raise ValueError("Model returned empty probability array")
        else:
            probs = probs.astype(float)
        return probs

    def _convert_probs_to_tags(self, probs: np.ndarray) -> list[str]:
        if (
            len(probs)
            < max(
                max(self.rating_indexes, default=-1),
                max(self.general_indexes, default=-1),
                max(self.character_indexes, default=-1),
            )
            + 1
        ):
            logger.error(
                f"Insufficient probabilities ({len(probs)}) for tag indices. Max rating index: {max(self.rating_indexes, default=-1)}, Max general index: {max(self.general_indexes, default=-1)}, Max char index: {max(self.character_indexes, default=-1)}"
            )
            raise ValueError(
                "Model returned insufficient probabilities for tag processing"
            )

        labels = list(zip(self.tag_names, probs, strict=False))

        # Handle cases where rating_indexes might be empty
        if self.rating_indexes:
            ratings_names = [
                labels[i] for i in self.rating_indexes if i < len(labels)
            ]
            ratings_dict = dict(ratings_names)
            top_rating = (
                max(ratings_dict.items(), key=lambda x: x[1])
                if ratings_dict
                else ("unknown", 0.0)
            )
            rating_tag = top_rating[0].replace("_", " ")
        else:
            rating_tag = (
                "unknown"  # Default if no rating tags are defined/found
            )

        # Use current_*_threshold as set by __init__ (which considers user, model-specific, defaults)
        # These are already attributes like self.general_threshold

        general_threshold_val = self.calculate_threshold(
            probs,
            self.general_indexes,
            self.use_mcut_general,
            self.general_threshold,  # Base threshold from __init__
            self.min_general_threshold,
        )
        character_threshold_val = self.calculate_threshold(
            probs,
            self.character_indexes,
            self.use_mcut_character,
            self.character_threshold,  # Base threshold from __init__
            self.min_character_threshold,
        )

        sorted_general_labels = self.process_tags(
            probs, self.general_indexes, general_threshold_val
        )
        sorted_character_labels = self.process_tags(
            probs, self.character_indexes, character_threshold_val
        )

        all_tags = []
        if (
            rating_tag != "unknown"
        ):  # Only add rating tag if it's not the default placeholder
            all_tags.append(rating_tag)
        all_tags += self.format_tags(sorted_character_labels)
        all_tags += self.format_tags(sorted_general_labels)
        return all_tags


class JoyTagBooruModel(BaseBooruModel):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_name: str = "JoyTag",  # JoyTag typically only has one model name in this context
        use_mcut_general: bool = False,  # JoyTag doesn't use character tags/thresholds
        # min_general_threshold, general_threshold are relevant via joytag_threshold
        min_general_threshold: float = None,
        general_threshold: float = None,  # Effectively joytag_threshold for JoyTag
        joytag_threshold: float = None,
        # Pass other common params even if not directly used by JoyTag specific logic, for super()
        use_mcut_character: bool = False,
        min_character_threshold: float = None,
        character_threshold: float = None,
    ):
        if model_name != "JoyTag":
            logger.warning(
                f"JoyTagBooruModel instantiated with model_name '{model_name}'. It will use 'JoyTag'."
            )

        # For JoyTag, general_threshold from params effectively becomes its main threshold if joytag_threshold is None
        # The base __init__ will handle setting self.joytag_threshold correctly based on passed joytag_threshold or default
        effective_joytag_thresh = (
            joytag_threshold
            if joytag_threshold is not None
            else general_threshold
        )

        super().__init__(
            device,
            dtype,
            "JoyTag",  # Fixed model_name for JoyTag
            use_mcut_general,
            use_mcut_character,
            min_general_threshold,
            min_character_threshold,
            general_threshold,
            character_threshold,  # Pass general_threshold for potential use by joytag_threshold logic
            effective_joytag_thresh,  # Pass the resolved joytag_threshold
        )
        logger.debug(
            f"Initializing JoyTagBooruModel with model_name='{self.model_name}'"
        )
        self._load_model_and_tags()

    def _load_model_and_tags(self):
        repo_id = self.MODEL_REPOS[
            self.model_name
        ]  # self.model_name is "JoyTag"
        logger.info(f"Loading {self.model_name} model from {repo_id}")

        try:
            model_path = huggingface_hub.hf_hub_download(
                repo_id, "model.onnx"
            )
        except Exception as e:
            logger.error(
                f"Error downloading model.onnx from {repo_id}: {e}"
            )
            raise

        if self.device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider"
                in onnxruntime.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        try:
            self.model = onnxruntime.InferenceSession(
                model_path, providers=providers
            )
        except Exception as e:
            logger.error(
                f"Error creating ONNX runtime session for model {self.model_name}: {e}"
            )
            raise

        self.input = self.model.get_inputs()[0]
        self.output = self.model.get_outputs()[0]
        self.input_shape = (
            self.input.shape
        )  # JoyTag expects [1, 3, height, width]

        if len(self.input_shape) == 4 and self.input_shape[1] == 3:  # NCHW
            _, _, self.input_height, self.input_width = self.input_shape
            logger.debug(
                f"Model expects input shape: {self.input_shape} (H: {self.input_height}, W: {self.input_width})"
            )
        else:
            logger.warning(
                f"Unexpected input shape for JoyTag: {self.input_shape}, using default dimensions 448x448"
            )
            self.input_height = self.input_width = (
                448  # Default from original code for unexpected shapes
            )

        self.tag_names = []
        self.general_indexes = []  # JoyTag considers all tags general
        # rating_indexes and character_indexes remain empty for JoyTag

        try:
            label_path = huggingface_hub.hf_hub_download(
                repo_id, "top_tags.txt"
            )
        except Exception as e:
            logger.error(
                f"Error downloading top_tags.txt from {repo_id}: {e}"
            )
            raise
        with open(label_path, "r", encoding="utf-8") as file:
            self.tag_names = [line.strip() for line in file.readlines()]
            self.general_indexes = list(range(len(self.tag_names)))
            logger.debug(f"Loaded {len(self.tag_names)} JoyTag tags")

    def preprocess_image_joytag(self, image: Image.Image) -> np.ndarray:
        """
        Special preprocessing for JoyTag following their official implementation

        Args:
            image: Input PIL image

        Returns:
            4D numpy array with shape [1, channels, height, width]
        """
        # Use common preprocessing for padding and resizing
        # Use self.input_height which is determined during model loading
        processed_image = self.preprocess_image_base(
            image, self.input_height, square_pad=True
        )  # JoyTag uses square_pad=True

        # JT specific normalization
        img_array = np.asarray(processed_image, dtype=np.float32) / 255.0
        mean = np.array(
            [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
        )
        std = np.array(
            [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
        )
        img_array = (img_array - mean) / std

        return img_array.transpose(2, 0, 1)[np.newaxis, :]

    def _preprocess_image_for_model(
        self, image: Image.Image
    ) -> np.ndarray:
        return self.preprocess_image_joytag(image)

    def _run_model_and_get_probs(
        self, input_tensor: np.ndarray
    ) -> np.ndarray:
        input_name = self.input.name
        label_name = self.output.name
        raw_output = self.model.run(
            [label_name], {input_name: input_tensor}
        )[0]

        if not isinstance(raw_output, np.ndarray) or raw_output.ndim < 1:
            logger.error(
                f"Unexpected output from JoyTag model: {type(raw_output)}"
            )
            raise ValueError(
                f"Model returned invalid output type: {type(raw_output)}"
            )

        # Sigmoid activation for JoyTag
        probs = 1.0 / (1.0 + np.exp(-raw_output))

        if probs.ndim == 0 or probs.shape[0] == 0:
            logger.error(
                f"Empty probability array from JoyTag model: {probs.shape}"
            )
            raise ValueError("Model returned empty probability array")

        # JoyTag output is often [[probs_for_tags]], so take the first element if nested
        if probs.ndim > 1 and probs.shape[0] == 1:
            return probs[0]
        return probs

    def _convert_probs_to_tags(self, probs: np.ndarray) -> list[str]:
        # JoyTag uses self.joytag_threshold, which is set in BaseBooruModel.__init__
        # It considers passed joytag_threshold, then DEFAULT_JOYTAG_THRESHOLD.
        # If MCut is enabled for general tags, it could also be used.

        current_joytag_threshold = self.joytag_threshold
        if self.use_mcut_general:  # JoyTag tags are all 'general'
            # For MCut, min_general_threshold is used as the floor
            mcut_based_threshold = self.calculate_threshold(
                probs,
                self.general_indexes,
                True,  # use_mcut = True
                self.joytag_threshold,  # base_threshold (fallback if mcut fails or not used)
                self.min_general_threshold,  # min_threshold for mcut
            )
            logger.debug(
                f"JoyTag MCut calculated threshold: {mcut_based_threshold}, original joytag_threshold: {self.joytag_threshold}"
            )
            current_joytag_threshold = mcut_based_threshold
        else:
            logger.debug(
                f"JoyTag using threshold: {current_joytag_threshold}"
            )

        sorted_labels = self.process_tags(
            probs, self.general_indexes, current_joytag_threshold
        )
        all_tags = self.format_tags(sorted_labels)
        return all_tags
