import logging
from typing import Any

from modules.module.captioning.BaseImageCaptionModel import (
    BaseImageCaptionModel,
    CaptionSample,
)
from modules.module.captioning.caption_config_types import (
    JoyCaptionGenerationConfig,
)

import torch

from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)

# Constants for the JoyCaption model
MODEL_PATH = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512

# Caption type prompts from the app
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        """Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.""",
        """Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.""",
        """Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.""",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

# Extra options choices from the app
NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."

EXTRA_OPTIONS = [
    NAME_OPTION,
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
    "If it is a work of art, do not include the artist's name or the title of the work.",
    "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
    """Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc.""",
    "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
    "Include information about the ages of any people/characters when applicable.",
    "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
    "Do not mention the mood/feeling/etc of the image.",
    "Explicitly specify the vantage height (eye-level, low-angle worm's-eye, bird's-eye, drone, rooftop, etc.).",
    "If there is a watermark, you must mention it.",
    """Your response will be used by a text-to-image model, so avoid useless meta phrases like "This image shows…", "You are looking at...", etc.""",
]

# Caption length choices
CAPTION_LENGTH_CHOICES = [
    "any",
    "very short",
    "short",
    "medium-length",
    "long",
    "very long",
] + [str(i) for i in range(20, 261, 10)]



class JoyCaptionModel(BaseImageCaptionModel):
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str | None = None,
        # Instance defaults are set from module constants.
        **loader_kwargs: Any,
    ):
        super().__init__()

        self.model_path = model_path
        # Instance defaults for generation parameters
        self.default_temperature: float = DEFAULT_TEMPERATURE
        self.default_top_p: float = DEFAULT_TOP_P
        self.default_max_tokens: int = DEFAULT_MAX_TOKENS
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing JoyCaptionModel '{self.model_path}' on device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Prepare kwargs for from_pretrained, ensuring type consistency
        effective_loader_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        effective_loader_kwargs.update(loader_kwargs) # Allow override via **loader_kwargs

        if self.device != "cpu" and "device_map" not in effective_loader_kwargs:
            effective_loader_kwargs["device_map"] = "auto"

        # If stream was passed in loader_kwargs and affects model loading, it's handled here.

        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            **effective_loader_kwargs
        )

        if self.device == "cpu" and "device_map" not in effective_loader_kwargs:
            self.llava_model.to(self.device)

        self.llava_model.eval()
        logger.info(f"JoyCaption model '{self.model_path}' loaded on device: {self.llava_model.device}")

    def generate_caption(
        self,
        caption_sample: CaptionSample,
        prompt: str, # The fully constructed prompt for the model
        generation_config: JoyCaptionGenerationConfig | None = None
    ) -> str:
        image = caption_sample.get_image()
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Determine generation parameters: use config if provided, else instance defaults
        cfg = generation_config or {}
        current_temperature = cfg.get('temperature', self.default_temperature)
        current_top_p = cfg.get('top_p', self.default_top_p)
        current_max_tokens = cfg.get('max_tokens', self.default_max_tokens)

        # The 'prompt' parameter is the fully formed prompt from the UI/caller
        llava_prompt = prompt

        with torch.no_grad():
            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
                },
                {
                    "role": "user",
                    "content": llava_prompt.strip(),
                },
            ]

            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            assert isinstance(convo_string, str)

            inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.llava_model.device)
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            generate_ids = self.llava_model.generate(
                **inputs,
                max_new_tokens=current_max_tokens,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=current_temperature,
                top_k=None, # JoyCaption typically uses top_p, not top_k
                top_p=current_top_p,
            )[0]

            input_ids_on_device = inputs['input_ids'].to(generate_ids.device)
            generate_ids = generate_ids[input_ids_on_device.shape[1]:]

            generated_text = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_text = generated_text.strip()

            # UI Prefix and postfix are applied externally by BaseImageCaptionModel.caption_image
            return generated_text
