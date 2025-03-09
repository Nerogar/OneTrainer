from modules.module.BaseImageCaptionModel import (
    BaseImageCaptionModel,
    CaptionSample,
)

import torch

from transformers import AutoModelForCausalLM


class Moondream2Model(BaseImageCaptionModel):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_revision: str = "2025-01-09",
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

        print(f"Initializing Moondream2Model with revision {model_revision}")

        device_map = {"": "cuda"} if device.type == "cuda" else None

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=model_revision,
            trust_remote_code=True,
            device_map=device_map,
        )

        print("Moondream2 model loaded successfully")

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        initial_caption: str = "",
        caption_prefix: str = "",
        caption_postfix: str = "",
    ) -> str:
        """
        Generate a caption for the given image using the Moondream2 model.

        Args:
            caption_sample: The image sample to caption
            initial_caption: Optional initial caption
            caption_prefix: Optional prefix to add to the caption
            caption_postfix: Optional postfix to add to the caption

        Returns:
            Generated caption
        """
        image = caption_sample.get_image()

        # Generate caption using Moondream2
        result = self.model.caption(
            image,
            length=self.caption_length,
            stream=self.stream
        )

        # Get the caption from the result
        generated_caption = result["caption"]

        # If streaming is enabled and the result is a generator, convert to string
        if self.stream and not isinstance(generated_caption, str):
            caption_text = ""
            for text_segment in generated_caption:
                caption_text += text_segment
            generated_caption = caption_text

        # Add initial caption if provided
        if initial_caption:
            generated_caption = f"{initial_caption} {generated_caption}"

        # Add prefix and postfix
        final_caption = (
            caption_prefix + generated_caption + caption_postfix
        ).strip()

        return final_caption
