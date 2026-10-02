"""Check a trained LoRA (and optionally its training images) for common causes of poor results.

Usage:
    python scripts/diagnose_lora.py <lora.safetensors> [image_folder]

Prints the training settings stored in the file (if any), how strongly the LoRA changed the
model, and - when an image folder is given or found in the stored settings - caption statistics
that often explain an inconsistent character (traits in every caption, no trigger word, ...).
"""
import json
import os
import re
import sys
from collections import Counter

from safetensors import safe_open

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jxl", ".avif"}

# caption tags describing a character's fixed traits. When these are in (almost) every caption the
# LoRA ties the look to the tags instead of learning it, so leaving them out of the prompt gives a
# random look (e.g. different hair colors).
TRAIT_PATTERN = re.compile(
    r"\b(hair|eyes?|bangs|ahoge|ponytail|twintails|braids?|sidelocks|pupils|ears|horns?|tail|wings|skin|breasts|freckles|mole)\b"
)


def section(title: str):
    print()
    print(f"== {title} ==")


def print_settings(metadata: dict[str, str]) -> dict | None:
    section("Stored metadata")
    print("architecture:", metadata.get("modelspec.architecture", "?"))
    print("date:", metadata.get("modelspec.date", "?"))
    raw = metadata.get("ot_config")
    if not raw:
        print("no training settings stored in this file (enable 'include train config' to keep them)")
        return None
    config = json.loads(raw)
    for key in ("base_model_name", "learning_rate", "epochs", "batch_size", "resolution",
                "lora_rank", "lora_alpha", "optimizer", "learning_rate_scheduler"):
        value = config.get(key)
        if isinstance(value, dict):
            value = value.get("optimizer", value)
        print(f"{key}: {value}")
    return config


def analyse_weights(path: str) -> int | None:
    section("LoRA weights")
    downs, ups, alphas = {}, {}, {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 (safe_open is not a dict)
            if key.endswith((".lora_down.weight", ".lora_A.weight")):
                downs[key.rsplit(".", 2)[0]] = f.get_tensor(key).float()
            elif key.endswith((".lora_up.weight", ".lora_B.weight")):
                ups[key.rsplit(".", 2)[0]] = f.get_tensor(key).float()
            elif key.endswith(".alpha"):
                alphas[key.rsplit(".", 1)[0]] = float(f.get_tensor(key))

    names = sorted(downs.keys() & ups.keys())
    if not names:
        print("no LoRA layers found (is this a full model or an embedding?)")
        return None

    ranks = Counter(downs[n].shape[0] for n in names)
    rank = ranks.most_common(1)[0][0]
    print(f"layers: {len(names)}, rank: {dict(ranks)}")

    # norm of the weight change each layer adds: |up @ down| * alpha / rank
    norms = []
    for n in names:
        down, up = downs[n].flatten(1), ups[n].flatten(1)
        scale = alphas.get(n, down.shape[0]) / down.shape[0]
        norms.append(float((up @ down).norm()) * scale)
    norms.sort()
    zero = sum(1 for x in norms if x < 1e-6)
    print(f"weight change per layer: median {norms[len(norms) // 2]:.4f}, max {norms[-1]:.4f}")
    if zero:
        print(f"WARNING: {zero} layers did not change at all")
    print("compare these numbers between the checkpoints of one training: if they barely grow, the LoRA is undertrained")
    return rank


def find_captions(folder: str) -> tuple[int, list[str]]:
    images, captions = 0, []
    for root, _, files in os.walk(folder):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            images += 1
            caption_path = os.path.join(root, stem + ".txt")
            if os.path.isfile(caption_path):
                with open(caption_path, encoding="utf-8", errors="replace") as f:
                    captions.append(f.read().strip())
    return images, captions


def analyse_dataset(folder: str):
    section(f"Training images: {folder}")
    if not os.path.isdir(folder):
        print("folder not found")
        return
    images, captions = find_captions(folder)
    print(f"images: {images}, with .txt caption: {len(captions)}")
    if images < 15:
        print("WARNING: few images - 20-50 varied images of the character usually work better")
    if not captions:
        return

    tags = Counter()
    for caption in captions:
        tags.update({t.strip().lower() for t in caption.replace("\n", ",").split(",") if t.strip()})

    print("most common tags:")
    for tag, count in tags.most_common(25):
        print(f"  {count / len(captions):5.0%}  {tag}")

    always = [tag for tag, count in tags.items() if count == len(captions)]
    if always:
        print("in every caption (good candidates for the trigger word):", ", ".join(always[:5]))
    else:
        print("WARNING: no tag is in every caption - add a unique trigger word (e.g. 'sopia') to all captions")

    traits = [tag for tag, count in tags.most_common()
              if count >= 0.5 * len(captions) and TRAIT_PATTERN.search(tag)]
    if traits:
        print("WARNING: fixed character traits in most captions:", ", ".join(traits[:10]))
        print("  remove them so the trigger word learns the look, or always put them in the prompt")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}

    config = print_settings(metadata)
    analyse_weights(path)

    folders = sys.argv[2:]
    if not folders and config:
        folders = [c["path"] for c in config.get("concepts") or [] if c.get("path")]
    for folder in folders:
        analyse_dataset(folder)


if __name__ == "__main__":
    main()
