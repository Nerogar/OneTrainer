import argparse
import json
import logging
import sys

import torch

from diffusers import DiffusionPipeline

# Stable substrings emitted by diffusers on key mismatches:
# - peft adapter loader (diffusers/utils/peft_utils.py:_maybe_warn_for_unhandled_keys)
# - state-dict format converters (diffusers/loaders/lora_conversion_utils.py), which
#   silently drop any key they can't map to a known format
UNEXPECTED_KEYS_MARKER = "led to unexpected keys found in the model"
MISSING_KEYS_MARKER = "led to missing keys in the model"
UNSUPPORTED_KEYS_MARKER = "Unsupported keys for"
# Emitted by diffusers/loaders/peft.py when a LoRA prefix matched zero keys in the
# state dict - i.e. that component's LoRA was silently a no-op. Benign for the
# text_encoder(_2) prefixes (OT presets often don't train a text-encoder LoRA), but
# the main denoiser (transformer/unet) getting zero keys means the LoRA didn't load at
# all, which would otherwise pass this script's checks above since no key name mismatch
# is reported in that case.
NO_KEYS_DENOISER_MARKERS = ("found with the prefix='transformer'", "found with the prefix='unet'")


class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset_path")
    parser.add_argument("lora_path")
    args = parser.parse_args()

    with open(args.preset_path, "r") as f:
        preset = json.load(f)
    base_model_name = preset["base_model_name"]

    handler = RecordingHandler()
    logging.getLogger("diffusers").addHandler(handler)

    pipe = DiffusionPipeline.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_path)

    unexpected = [m for m in handler.records if UNEXPECTED_KEYS_MARKER in m]
    missing = [m for m in handler.records if MISSING_KEYS_MARKER in m]
    unsupported = [m for m in handler.records if UNSUPPORTED_KEYS_MARKER in m]
    no_keys_denoiser = [m for m in handler.records if any(marker in m for marker in NO_KEYS_DENOISER_MARKERS)]

    if unexpected or missing or unsupported or no_keys_denoiser:
        for m in unexpected + missing + unsupported + no_keys_denoiser:
            print(f"FAIL: {m}", file=sys.stderr)
        sys.exit(1)

    print("OK: LoRA loaded into diffusers pipeline with no unexpected/missing adapter keys.")


if __name__ == '__main__':
    main()
