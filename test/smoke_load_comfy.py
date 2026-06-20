import argparse
import logging
import sys

# Loader to use, and (for "split") the comfy.sd.CLIPType to pass to load_clip.
# clip_type is None where Comfy auto-detects the text encoder from its weights
# (ERNIE, Z_IMAGE) or where the checkpoint loader handles clip internally.
MODEL_CONFIG = {
    "CHROMA_1": ("split", "CHROMA"),
    "ERNIE": ("split", None),
    "FLUX_DEV_1": ("split", "FLUX"),
    "FLUX_2": ("split", "FLUX2"),
    "HI_DREAM_FULL": ("split", "HIDREAM"),
    "HUNYUAN_VIDEO": ("split", "HUNYUAN_VIDEO"),
    "PIXART_SIGMA": ("split", "PIXART"),
    "QWEN": ("split", "QWEN_IMAGE"),
    "STABLE_DIFFUSION_15": ("checkpoint", None),
    "STABLE_DIFFUSION_21": ("checkpoint", None),
    "STABLE_DIFFUSION_3": ("checkpoint", None),
    "STABLE_DIFFUSION_XL_10_BASE": ("checkpoint", None),
    "Z_IMAGE": ("split", None),
}


class RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comfyui_path", help="Path to the ComfyUI checkout (for importing the comfy package)")
    parser.add_argument("model_type", choices=MODEL_CONFIG.keys())
    parser.add_argument("lora_path")
    parser.add_argument(
        "model_files", nargs="+",
        help="unet/checkpoint path, followed by any text-encoder paths (split loader only)",
    )
    args = parser.parse_args()

    loader, clip_type_name = MODEL_CONFIG[args.model_type]

    sys.path.insert(0, args.comfyui_path)

    import comfy.cli_args
    comfy.cli_args.args.cpu = True

    import comfy.sd
    import comfy.utils

    handler = RecordingHandler()
    logging.getLogger().addHandler(handler)

    if loader == "checkpoint":
        model, clip, _vae = comfy.sd.load_checkpoint_guess_config(args.model_files[0], output_clipvision=False)[:3]
    else:
        clip_type = getattr(comfy.sd.CLIPType, clip_type_name) if clip_type_name else comfy.sd.CLIPType.STABLE_DIFFUSION
        model = comfy.sd.load_diffusion_model(args.model_files[0])
        clip = comfy.sd.load_clip(ckpt_paths=args.model_files[1:], clip_type=clip_type)

    lora_sd = comfy.utils.load_torch_file(args.lora_path, safe_load=True)

    comfy.sd.load_lora_for_models(model, clip, lora_sd, 1.0, 1.0)

    not_loaded = [m for m in handler.records if "lora key not loaded" in m]

    if not_loaded:
        for m in not_loaded:
            print(f"FAIL: {m}", file=sys.stderr)
        sys.exit(1)

    print("OK: LoRA loaded onto Comfy model/clip with no unmatched keys.")


if __name__ == '__main__':
    main()
