from typing import NotRequired, TypedDict


class BaseGenerationConfig(TypedDict, total=False):
    """Base for generation configurations. All keys are optional."""


class JoyCaptionGenerationConfig(BaseGenerationConfig):
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    max_tokens: NotRequired[int]


class MoondreamGenerationConfig(BaseGenerationConfig):
    caption_length: NotRequired[str]


class WDGenerationConfig(BaseGenerationConfig):
    pass


class BlipGenerationConfig(BaseGenerationConfig):
    pass


# Union type for convenience, though `Any` will often be used in base class signatures
AllGenerationConfigs = (
    JoyCaptionGenerationConfig
    | MoondreamGenerationConfig
    | WDGenerationConfig
    | BlipGenerationConfig
)
