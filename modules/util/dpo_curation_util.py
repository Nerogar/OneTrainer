import json
import os
import random
import shutil

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_metadata_util import strip_angle_bracket_segments
from modules.util.path_util import supported_image_extensions


def is_dpo_concept_type(concept_type: ConceptType) -> bool:
    return concept_type in {
        ConceptType.DPO_CHOSEN,
        ConceptType.DPO_REJECTED,
        ConceptType.DPO_CHOSEN_VAL,
        ConceptType.DPO_REJECTED_VAL,
    }


def dpo_concept_pairs(concepts: list[ConceptConfig], is_validation: bool = False) -> list[tuple[str, str]]:
    enabled = [concept for concept in concepts if concept.enabled]
    chosen_type = ConceptType.DPO_CHOSEN_VAL if is_validation else ConceptType.DPO_CHOSEN
    rejected_type = ConceptType.DPO_REJECTED_VAL if is_validation else ConceptType.DPO_REJECTED
    chosen = [concept for concept in enabled if ConceptType(concept.type) == chosen_type]
    rejected = [concept for concept in enabled if ConceptType(concept.type) == rejected_type]

    if not chosen and not rejected:
        raise RuntimeError(
            f"Need explicit {chosen_type.value}/{rejected_type.value} concepts for RLHF DPO pairs."
        )
    if len(chosen) != len(rejected):
        raise RuntimeError(
            f"Mismatched DPO concept counts: {len(chosen)} chosen, {len(rejected)} rejected."
        )

    return [(chosen_concept.path, rejected_concept.path) for chosen_concept, rejected_concept in zip(chosen, rejected, strict=True)]


def has_existing_exports(output_dir: str) -> bool:
    for subdir in ("chosen", "rejected"):
        d = os.path.join(output_dir, subdir)
        if os.path.isdir(d):
            for _root, _dirs, files in os.walk(d):
                if any(file_name.startswith("pair_") for file_name in files):
                    return True
    return False


def export_curated_pairs(
    groups: list[dict],
    results: dict[int, list[dict]],
    output_dir: str,
    val_percentage: float = 0.0,
) -> tuple[str, str, str, str, int, int, int]:
    chosen_train_dir = os.path.join(output_dir, "chosen", "train")
    chosen_val_dir = os.path.join(output_dir, "chosen", "val")
    rejected_train_dir = os.path.join(output_dir, "rejected", "train")
    rejected_val_dir = os.path.join(output_dir, "rejected", "val")
    for d in (chosen_train_dir, chosen_val_dir, rejected_train_dir, rejected_val_dir):
        os.makedirs(d, exist_ok=True)

    group_indices = list(results.keys())
    random.shuffle(group_indices)
    val_count_target = int(len(group_indices) * (val_percentage / 100.0))
    val_groups = set(group_indices[:val_count_target])

    skipped_count = len(groups) - len(results)
    val_count = 0
    train_count = 0

    for group_idx, pairs in results.items():
        group = groups[group_idx]
        caption = strip_angle_bracket_segments(group["prompt"])
        is_val = group_idx in val_groups

        if is_val:
            chosen_dir, rejected_dir = chosen_val_dir, rejected_val_dir
        else:
            chosen_dir, rejected_dir = chosen_train_dir, rejected_train_dir

        for pair_idx, pair in enumerate(pairs):
            safe_name = f"pair_{group_idx:04d}_{pair_idx:04d}"
            chosen_ext = os.path.splitext(pair["chosen"])[1].lower()
            rejected_ext = os.path.splitext(pair["rejected"])[1].lower()
            _copy_image(pair["chosen"], os.path.join(chosen_dir, safe_name + chosen_ext))
            _copy_image(pair["rejected"], os.path.join(rejected_dir, safe_name + rejected_ext))
            for subdir in (chosen_dir, rejected_dir):
                with open(os.path.join(subdir, safe_name + ".txt"), "w", encoding="utf-8") as f:
                    f.write(caption)

        if is_val:
            val_count += len(pairs)
        else:
            train_count += len(pairs)

    abs_output = os.path.abspath(output_dir)
    concept_entries = [
        (os.path.join(abs_output, "chosen", "train"), ConceptType.DPO_CHOSEN),
        (os.path.join(abs_output, "rejected", "train"), ConceptType.DPO_REJECTED),
        (os.path.join(abs_output, "chosen", "val"), ConceptType.DPO_CHOSEN_VAL),
        (os.path.join(abs_output, "rejected", "val"), ConceptType.DPO_REJECTED_VAL),
    ]
    concepts = []
    for path, concept_type in concept_entries:
        cfg = ConceptConfig.default_values()
        cfg.path = path
        cfg.type = concept_type
        cfg.enabled = True
        concepts.append(cfg.to_dict())
    with open(os.path.join(abs_output, "concepts.json"), "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)

    return (chosen_train_dir, rejected_train_dir, chosen_val_dir, rejected_val_dir,
            skipped_count, val_count, train_count)


def dpo_pair_key(image_path: str, concept_path: str) -> str:
    try:
        relative = os.path.relpath(image_path, concept_path)
    except ValueError:
        relative = os.path.basename(image_path)
    return os.path.splitext(relative.replace('\\', '/'))[0]


def check_dpo_pairs(concept_pairs: list[tuple[str, str]]) -> dict:
    exts = supported_image_extensions()
    all_matched = 0
    all_chosen_stray = 0
    all_rejected_stray = 0
    format_stats: dict[str, int] = {}
    pairs_info = []

    for chosen_path, rejected_path in concept_pairs:
        chosen_keys: dict[str, str] = {}
        rejected_keys: dict[str, str] = {}

        for root, _dirs, files in os.walk(chosen_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in exts:
                    full = os.path.join(root, fname)
                    chosen_keys[dpo_pair_key(full, chosen_path)] = full

        for root, _dirs, files in os.walk(rejected_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in exts:
                    full = os.path.join(root, fname)
                    rejected_keys[dpo_pair_key(full, rejected_path)] = full

        matched_keys = set(chosen_keys) & set(rejected_keys)
        chosen_stray = len(chosen_keys) - len(matched_keys)
        rejected_stray = len(rejected_keys) - len(matched_keys)

        for key in matched_keys:
            for img_path in (chosen_keys[key], rejected_keys[key]):
                ext = os.path.splitext(img_path)[1].lower().lstrip('.')
                format_stats[ext] = format_stats.get(ext, 0) + 1

        all_matched += len(matched_keys)
        all_chosen_stray += chosen_stray
        all_rejected_stray += rejected_stray
        pairs_info.append({
            'chosen_path': chosen_path,
            'rejected_path': rejected_path,
            'matched': len(matched_keys),
            'chosen_stray': chosen_stray,
            'rejected_stray': rejected_stray,
        })

    return {
        'total_matched': all_matched,
        'total_chosen_stray': all_chosen_stray,
        'total_rejected_stray': all_rejected_stray,
        'format_stats': format_stats,
        'pairs': pairs_info,
    }


def _copy_image(source_path: str, target_path: str):
    shutil.copy2(source_path, target_path)
