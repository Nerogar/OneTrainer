import logging
import traceback

from modules.module.captioning.BaseImageCaptionModel import (
    BaseImageCaptionModel,
)
from modules.module.captioning.CaptionSample import CaptionSample

import torch

from transformers import AutoModelForCausalLM

from PIL import Image

from .captioning.caption_config_types import MoondreamGenerationConfig

logger = logging.getLogger(__name__)

class MoondreamModel(BaseImageCaptionModel):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_revision: str = "4f5917588ff780f0ae93c1b15a0e39afa5139b8f", #DO NOT CHANGE. Commit hash has been reviewed.
        caption_length: str = "normal",
        stream: bool = False, # Default stream value for the model instance
    ):
        """
        Initialize the Moondream2 captioning model.

        Args:
            device (torch.device): Device to use (CPU/CUDA)
            dtype (torch.dtype): Data type for model
            model_revision (str): Model revision to use
            caption_length (str): "short", "normal", or "long" caption length
            stream (bool): Whether the underlying model call should stream output.
        """
        self.device = device
        self.dtype = dtype
        self.caption_length = caption_length
        self.stream = stream # This instance's stream preference

        device_map = {"": "cuda"} if device.type == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=model_revision,
            trust_remote_code=True,
            device_map=device_map,
        )
        self.model.eval()

    def _get_image(self, caption_sample: CaptionSample) -> Image.Image | None:
        """Helper method to get the image from the caption sample."""
        try:
            image = caption_sample.get_image()
            # logger.debug("MoondreamModel: Acquired image") # Can be verbose
            return image
        except Exception:
            logger.exception("MoondreamModel: ERROR getting image from caption_sample")
            return None # Return None, not empty string

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        prompt: str,  # Used for interface consistency, Moondream might not use it directly
        generation_config: MoondreamGenerationConfig | None = None
    ) -> str:
        current_caption_length = self.caption_length  # Start with instance default
        if generation_config and generation_config.get('caption_length') is not None:
            current_caption_length = generation_config['caption_length']

        logger.debug(
            f"MoondreamModel: Generating caption. Effective Length: {current_caption_length}, Stream setting for model call: {self.stream}. "
            f"Prompt (interface): '{prompt}'"
        )
        image = self._get_image(caption_sample)
        if image is None:
            logger.warning("MoondreamModel: Could not get image from sample, returning empty caption.")
            return ""

        try:
            with torch.no_grad():
                # Call the underlying moondream2 model's caption method
                model_output_dict = self.model.caption(
                    image,
                    length=current_caption_length, # Use effective length
                    stream=self.stream
                )

                raw_caption_content: any
                if isinstance(model_output_dict, dict):
                    raw_caption_content = model_output_dict.get("caption", "")
                else:
                    logger.warning(f"MoondreamModel: model.caption did not return a dict. Got: {type(model_output_dict)}. Using raw output if string.")
                    raw_caption_content = str(model_output_dict) if isinstance(model_output_dict, str) else ""

                # Process the raw caption content
                assembled_caption: str
                if hasattr(raw_caption_content, '__iter__') and not isinstance(raw_caption_content, str):
                    tokens = list(raw_caption_content)  # Fully consume any generator/iterator
                    assembled_caption = "".join(str(t) for t in tokens)
                elif isinstance(raw_caption_content, str):
                    assembled_caption = raw_caption_content
                else:
                    assembled_caption = ""

                # Replace all newlines with spaces
                assembled_caption = assembled_caption.replace("\n", " ")

                final_caption = assembled_caption.strip()
                return final_caption

        except Exception as e:
            logger.error(f"ERROR in MoondreamModel.generate_caption: {str(e)}")
            traceback.print_exc()
            return ""

    def generate_detection(
            self,
            caption_sample: CaptionSample,
            prompt: str,
        ) -> dict[str, list[dict[str, list[float] | float]]]:
        """
        Detect objects in an image using Moondream2's detection capability.

        Args:
            caption_sample (CaptionSample): The sample containing the image
            prompt (str): The type of object to detect (e.g., "face", "person"), more lengthy prompts can work but are less reliable

        Returns:
            dict: Detection results with objects and their bounding boxes
                Format: {"objects": [{"box": [x0, y0, x1, y1], "score": float}]}

            Where:
            - x0, y0: Top-left corner coordinates (normalized to image dimensions)
            - x1, y1: Bottom-right corner coordinates (normalized to image dimensions)
            - score: Detection confidence (higher is better)

            Returns empty objects list on failure: {"objects": []}
        """
        logger.debug(f"DEBUG START: Generating detection for prompt: {prompt}")

        image = self._get_image(caption_sample)
        if image is None:
            return {"objects": []}

        # Check if model has the detect attribute
        if not hasattr(self.model, 'detect'):
            logger.error("ERROR: Moondream2 model does not have 'detect' method. Check model version.")
            return {"objects": []}

        logger.debug("DEBUG: About to call model.detect")

        # Generate detection using Moondream2 - with timeout protection
        try:
            with torch.no_grad():
                result = self.model.detect(image, prompt)
                logger.debug(f"DEBUG: Detection completed. Raw result type: {type(result)}")

                # Ensure we have a proper dictionary structure
                if not isinstance(result, dict):
                    logger.warning(f"WARNING: Unexpected result type: {type(result)}")
                    result = {"objects": []}

                if "objects" not in result:
                    logger.warning("WARNING: 'objects' key missing from detection result")
                    result["objects"] = []

                logger.debug(f"DEBUG END: Detection result contains {len(result.get('objects', []))} objects")
                return result
        except Exception as e:
            logger.error(f"ERROR in model.detect: {str(e)}")
            traceback.print_exc()
            return {"objects": []}
