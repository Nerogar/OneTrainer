from enum import Enum


class BaseEnum(Enum):
    def __str__(self):
        return self.value

    def pretty_print(self):
        # TODO: do we want this method to use translatable strings? If so, how to avoid introducing an undesirable QT dependency in modules.util.enum?
        return self.value.replace("_", " ").title()

    @staticmethod
    def is_enabled(value, context=None):
        return True

    @classmethod
    def enabled_values(cls, context=None):
        return [v for v in cls if cls.is_enabled(v, context)]

# CONTEXTS:
# DataType: embeddings, lora, [MODELCONTROLLERCOMPLEX], training_fallback, training_dtype, convert
# ModelFormat: [MODELCONTROLLERCOMPLEX], convert
# NoiseScheduler: [SAMPLECONTROLLER: one context, is_enabled overridden]
# ConceptType: no_prior_pred, prior_pred
# ModelType: COMPARE CONVERT VS MAIN
# TrainingMethod: convert, [MAINCONTROLLERCOMPLEXBEHAVIOR]

# Tabs:
# DataType -> Embeddings: Float32, BFloat16; Lora: Float32, BFloat16; Model: [COMPLEX BEHAVIOR], Training: [FALLBACK: Float32, BFloat16], [TRAINDTYPE: Float32, Float16, BFloat16, TFloat32]
# ConfigPart -> Model: None, Settings, All (always true)
# ModelFormat -> Model: [COMPLEX BEHAVIOR]

# Widgets:
# NoiseScheduler: Sample: DDIM, Euler, Euler A, UniPC, Euler Karras, DPM++ Karras, DPM++ SDE Karras

# Windows:
# ConceptType -> Concept: always true, except prior prediction only with lora training
# ModelType -> Convert: [ LONG LIST ], MAIN: [ LONG LIST ]
# TrainingMethod -> Convert: Fine tune, Lora, Embedding, MAIN: [COMPLEX BEHAVIOR]
# DataType -> Convert: Float32, Float16, BFloat16
# ModelFormat -> Convert: Safetensors, Diffusers
