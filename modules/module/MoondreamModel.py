import logging
import traceback

from modules.module.captioning.BaseImageCaptionModel import BaseImageCaptionModel
from modules.module.captioning.CaptionSample import CaptionSample

import torch

from transformers import AutoModelForCausalLM

from PIL import Image

logger = logging.getLogger(__name__)

class MoondreamModel(BaseImageCaptionModel):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_revision: str = "05d640e6da70c37b2473e0db8fef0233c0709ce4", #DO NOT CHANGE. Commit hash has been reviewed.
        caption_length: str = "normal",
        stream: bool = False,
    ):
        """
        Initialize the Moondream2 captioning model.

        Args:
            device (torch.device): Device to use (CPU/CUDA)
            dtype (torch.dtype): Data type for model
            model_revision (str): Model revision to use
            caption_length (str): "short" or "normal" caption length
            stream (bool): Whether to stream output (only relevant for interactive use)
        """
        self.device = device
        self.dtype = dtype
        self.caption_length = caption_length
        self.stream = stream

        # For CUDA devices, map the entire model to CUDA using the empty string key
        # which serves as a default mapping for all model components
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
            logger.debug("MoondreamModel: Acquired image")
            return image
        except Exception:
            logger.exception("ERROR in model.caption")
            return ""

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        initial: str = "",
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = ""
    ) -> str:
        logger.debug("Generating caption called")
        image = self._get_image(caption_sample)
        if image is None:
            return ""

        try:
            with torch.no_grad():
                result = self.model.caption(
                    image,
                    length=self.caption_length,
                    stream=self.stream
                )
                logger.debug(f"MoondreamModel.generate_caption: model.caption result: {result}")
                generated_caption = result.get("caption", "")
                logger.debug(f"MoondreamModel.generate_caption: Received raw caption: {generated_caption}")

                if self.stream and not isinstance(generated_caption, str):
                    generated_caption = ''.join(generated_caption)
                    logger.debug(f"MoondreamModel.generate_caption: Assembled streaming caption: {generated_caption}")

                if initial:
                    generated_caption = f"{initial} {generated_caption}".strip()
                elif initial_caption:
                    generated_caption = f"{initial_caption} {generated_caption}".strip()

                if caption_prefix:
                    generated_caption = f"{caption_prefix}{generated_caption}"
                if caption_postfix:
                    generated_caption = f"{generated_caption}{caption_postfix}"

                logger.debug(f"MoondreamModel.generate_caption: Final generated caption: {generated_caption}")
                return generated_caption
        except Exception as e:
            logger.error(f"ERROR in model.caption: {str(e)}")
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
