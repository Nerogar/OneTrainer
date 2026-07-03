import math
import os
from fractions import Fraction

from modules.util.path_util import supported_image_extensions, walk_skipping_dotted

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import numpy as np
from PIL import Image

# Quantization per model, cross-checked against modules/dataLoader/*BaseDataLoader.py
MODEL_QUANTIZATION = {
    "Z_IMAGE": 64,
    "CHROMA_1": 64,
    "QWEN": 64,
    "FLUX_DEV_1": 64,
    "FLUX_FILL_DEV_1": 64,
    "FLUX_2": 64,
    "HUNYUAN_VIDEO": 64,
    "STABLE_DIFFUSION_XL_10_BASE": 64,
    "STABLE_DIFFUSION_XL_10_BASE_INPAINTING": 64,
    "STABLE_DIFFUSION_15": 8,
    "STABLE_DIFFUSION_15_INPAINTING": 8,
    "STABLE_DIFFUSION_20": 8,
    "STABLE_DIFFUSION_20_BASE": 8,
    "STABLE_DIFFUSION_20_INPAINTING": 8,
    "STABLE_DIFFUSION_20_DEPTH": 8,
    "STABLE_DIFFUSION_21": 8,
    "STABLE_DIFFUSION_21_BASE": 8,
    "STABLE_DIFFUSION_3": 64,
    "STABLE_DIFFUSION_35": 64,
    "HI_DREAM_FULL": 64,
    "SANA": 32,
    "PIXART_ALPHA": 16,
    "PIXART_SIGMA": 16,
    "WUERSTCHEN_2": 128,
    "STABLE_CASCADE_1": 128,
}


def quantization_for_model(model_type: str) -> int:
    return MODEL_QUANTIZATION.get(str(model_type), 64)


def _quantize_res(res, q: int):
    return (round(res[0] / q) * q, round(res[1] / q) * q)


def build_buckets(target_resolution: int, q: int):
    """Build bucket resolutions + aspect ratios for a single target resolution.

    Mirrors venv/src/mgds/src/mgds/pipelineModules/AspectBucketing.py.
    Returns (list of (h, w) buckets, numpy array of h/w ratios).
    """
    new = [
        (
            h / math.sqrt(h * w) * target_resolution,
            w / math.sqrt(h * w) * target_resolution,
        )
        for (h, w) in AspectBucketing.all_possible_input_aspects
    ]
    new = new + [(w, h) for (h, w) in new]
    new = [_quantize_res(r, q) for r in new]
    new = list(set(new))
    aspects = np.array([h / w for (h, w) in new])
    return new, aspects


def assign_bucket(h: int, w: int, target_resolution: int, q: int):
    buckets, aspects = build_buckets(target_resolution, q)
    idx = int(np.argmin(np.abs(aspects - (h / w))))
    return buckets[idx]


_STANDARD_RATIOS = [
    # (h, w, label)
    (1, 1, "1:1 square"),
    (2, 3, "2:3 portrait"),
    (3, 2, "3:2 landscape"),
    (3, 4, "3:4 portrait"),
    (4, 3, "4:3 landscape"),
    (4, 5, "4:5 portrait"),
    (5, 4, "5:4 landscape"),
    (9, 16, "9:16 tall portrait"),
    (16, 9, "16:9 wide landscape"),
    (1, 2, "1:2 tall portrait"),
    (2, 1, "2:1 wide landscape"),
    (7, 4, "7:4 wide landscape"),
    (4, 7, "4:7 tall portrait"),
    (5, 8, "5:8 tall portrait"),
    (8, 5, "8:5 wide landscape"),
    (1, 3, "1:3 very tall"),
    (3, 1, "3:1 very wide"),
    (1, 4, "1:4 very tall"),
    (4, 1, "4:1 very wide"),
]


def label_aspect(h: int, w: int) -> str:
    """Human-readable aspect ratio label. Matches within ~3% to a standard name,
    otherwise falls back to the reduced fraction H:W with orientation.
    """
    if h <= 0 or w <= 0:
        return "?"
    ratio = h / w
    for sh, sw, name in _STANDARD_RATIOS:
        std = sh / sw
        if abs(ratio - std) / std <= 0.03:
            if sh == sw:
                return name
            if abs(ratio - std) < 1e-6:
                return name
            return "~" + name
    frac = Fraction(h, w).limit_denominator(64)
    if frac.numerator > frac.denominator:
        orient = "landscape"
    elif frac.numerator < frac.denominator:
        orient = "portrait"
    else:
        orient = "square"
    return f"~{frac.numerator}:{frac.denominator} {orient}"


_DEFAULT_EXCLUDE_POSTFIX = ("-masklabel", "-condlabel")


def _iter_image_files(
    concept_path: str,
    include_subdirectories: bool = True,
    exclude_postfix: tuple[str, ...] = _DEFAULT_EXCLUDE_POSTFIX,
):
    """Yield image paths under concept_path using OneTrainer's CollectPaths rules:

    - Skip subdirectories whose basename starts with '.'
    - Filter to supported image extensions
    - Exclude filenames whose stem ends with any of `exclude_postfix`
      (defaults to '-masklabel' and '-condlabel', matching the data loader)
    - Recurse only when `include_subdirectories` is True
    """
    exts = supported_image_extensions()

    def _matches(fname: str) -> bool:
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            return False
        return not any(stem.endswith(pf) for pf in exclude_postfix)

    for root, files in walk_skipping_dotted(concept_path, include_subdirectories):
        for fname in files:
            if _matches(fname):
                yield os.path.join(root, fname)


def _read_image_size(path: str):
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def analyze_concept(
    concept_path: str,
    batch_size: int,
    target_resolutions: list[int],
    quantization: int,
    include_subdirectories: bool = True,
) -> dict:
    """Bucket every image under concept_path and compute drop/add/remove counts
    per bucket for the given batch_size, for each target resolution.

    Image discovery mirrors the trainer's CollectPaths pipeline module: dotted
    subdirectories are skipped, '-masklabel' / '-condlabel' files are excluded,
    and recursion follows the per-concept `include_subdirectories` flag.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not target_resolutions:
        raise ValueError("target_resolutions must contain at least one value")
    if quantization <= 0:
        raise ValueError("quantization must be a positive integer")
    if not os.path.isdir(concept_path):
        raise ValueError(f"concept_path is not a directory: {concept_path}")

    # Cache image dims once across all target passes.
    scanned = 0
    unreadable = 0
    dims: list[tuple[int, int]] = []  # list of (h, w)
    for img_path in _iter_image_files(concept_path, include_subdirectories=include_subdirectories):
        scanned += 1
        size = _read_image_size(img_path)
        if size is None:
            unreadable += 1
            continue
        w, h = size
        dims.append((h, w))

    targets_out = []
    for target in target_resolutions:
        buckets, aspects = build_buckets(int(target), int(quantization))
        counts: dict[tuple[int, int], int] = {tuple(b): 0 for b in buckets}
        for h, w in dims:
            idx = int(np.argmin(np.abs(aspects - (h / w))))
            b = buckets[idx]
            counts[b] = counts.get(b, 0) + 1

        bucket_rows = []
        total_pairs = 0
        total_drops = 0
        total_add = 0
        total_remove = 0
        for b, count in counts.items():
            if count == 0:
                continue
            bh, bw = int(b[0]), int(b[1])
            drops = count % batch_size
            add = (batch_size - count % batch_size) % batch_size
            remove = drops
            bucket_rows.append(
                {
                    "h": bh,
                    "w": bw,
                    "count": count,
                    "drops": drops,
                    "add": add,
                    "remove": remove,
                    "aspect_label": label_aspect(bh, bw),
                }
            )
            total_pairs += count
            total_drops += drops
            total_add += add
            total_remove += remove

        bucket_rows.sort(key=lambda row: row["count"], reverse=True)

        targets_out.append(
            {
                "target": int(target),
                "total_pairs": total_pairs,
                "total_drops": total_drops,
                "total_add": total_add,
                "total_remove": total_remove,
                "buckets": bucket_rows,
            }
        )

    return {
        "concept_path": concept_path,
        "batch_size": int(batch_size),
        "quantization": int(quantization),
        "scanned": scanned,
        "unreadable": unreadable,
        "targets": targets_out,
    }


def parse_target_resolutions(resolution_str: str) -> list[int]:
    """Parse a TrainConfig.resolution string into a list of int targets.

    Handles '512', '512,768', '512, 768'. Silently skips fixed WxH forms
    (those bypass the analyzer since they ignore aspect bucketing).
    """
    if not resolution_str:
        return []
    out: list[int] = []
    for token in resolution_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" in token.lower():
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out
