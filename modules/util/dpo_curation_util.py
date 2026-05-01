import json
import os
import random
import re
import shutil

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_metadata_util import strip_angle_bracket_segments
from modules.util.path_util import supported_image_extensions

UNCONDITIONAL_PROMPT = "UNCONDITIONAL"


def _has_meaningful_content(prompt: str) -> bool:
    """A prompt is 'meaningful' only if at least one alphanumeric character or
    CJK/letter-like glyph survives. Stray punctuation (commas, dots, dashes)
    left behind by bracket stripping does not count — such prompts produce no
    real textual guidance and should be treated as unconditional."""
    return bool(re.search(r"[^\W\d_]|\d", prompt))


def normalize_prompt_for_grouping(prompt: str) -> str:
    """Return the cleaned prompt or UNCONDITIONAL when nothing meaningful
    survives bracket stripping. Use this everywhere we group/key by prompt.

    Stripping is iterated so nested/malformed brackets like ``<a<b>>`` collapse
    fully — a single regex pass removes only the inner ``<b>`` and leaves
    ``<a>`` behind, which the next pass also removes."""
    cleaned = prompt or ""
    while True:
        next_cleaned = strip_angle_bracket_segments(cleaned)
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    if _has_meaningful_content(cleaned):
        return cleaned
    return UNCONDITIONAL_PROMPT


def _normalize_source_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.normcase(os.path.abspath(path))
    except (OSError, ValueError):
        return path


def manifest_used_sources(manifest: dict) -> set[str]:
    """Set of source-image paths recorded in any manifest entry. Use this to
    avoid re-presenting an image that was already committed as chosen or
    rejected — preventing the same file ending up on both sides of a pair."""
    used: set[str] = set()
    for entry in manifest.get("pairs", []):
        for key in ("chosen_source", "rejected_source"):
            normed = _normalize_source_path(entry.get(key))
            if normed:
                used.add(normed)
    return used


def is_source_used(used_sources: set[str], path: str) -> bool:
    normed = _normalize_source_path(path)
    return normed is not None and normed in used_sources


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


def load_manifest(output_dir: str) -> dict:
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"pairs": []}


def save_manifest(output_dir: str, manifest: dict):
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def manifest_pair_counts(manifest: dict) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for entry in manifest.get("pairs", []):
        key = (normalize_prompt_for_grouping(entry["prompt"]), entry.get("aspectratio", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _next_pair_id(manifest: dict) -> int:
    pairs = manifest.get("pairs", [])
    if not pairs:
        return 0
    return max(entry["pair_id"] for entry in pairs) + 1


def export_single_pair(
    output_dir: str,
    manifest: dict,
    chosen_path: str,
    rejected_path: str,
    prompt: str,
    aspectratio: str,
):
    chosen_dir = os.path.join(output_dir, "chosen")
    rejected_dir = os.path.join(output_dir, "rejected")
    os.makedirs(chosen_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    pair_id = _next_pair_id(manifest)
    safe_name = f"pair_{pair_id:04d}"
    caption = strip_angle_bracket_segments(prompt)

    chosen_ext = os.path.splitext(chosen_path)[1].lower()
    rejected_ext = os.path.splitext(rejected_path)[1].lower()

    _copy_image(chosen_path, os.path.join(chosen_dir, safe_name + chosen_ext))
    _copy_image(rejected_path, os.path.join(rejected_dir, safe_name + rejected_ext))
    for subdir in (chosen_dir, rejected_dir):
        with open(os.path.join(subdir, safe_name + ".txt"), "w", encoding="utf-8") as f:
            f.write(caption)

    manifest.setdefault("pairs", []).append({
        "pair_id": pair_id,
        "prompt": prompt,
        "aspectratio": aspectratio,
        "chosen_file": safe_name + chosen_ext,
        "rejected_file": safe_name + rejected_ext,
        # Source paths let us filter already-used images out of future groups,
        # so the same file can't be picked again on the opposite side of a
        # pair (especially relevant for unconditional groups, which have no
        # pairs-per-group cap to skip them on resume).
        "chosen_source": _normalize_source_path(chosen_path),
        "rejected_source": _normalize_source_path(rejected_path),
    })
    save_manifest(output_dir, manifest)


def find_exported_file(base_dir: str, filename: str) -> str | None:
    for subdir in ("", "train", "val"):
        path = os.path.join(base_dir, subdir, filename) if subdir else os.path.join(base_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def find_orphaned_pairs(output_dir: str, manifest: dict) -> list[dict]:
    chosen_dir = os.path.join(output_dir, "chosen")
    rejected_dir = os.path.join(output_dir, "rejected")
    orphans = []
    for entry in manifest.get("pairs", []):
        chosen = find_exported_file(chosen_dir, entry["chosen_file"])
        rejected = find_exported_file(rejected_dir, entry["rejected_file"])
        if not chosen or not rejected:
            orphans.append(entry)
    return orphans


def prune_orphaned_pairs(output_dir: str, manifest: dict) -> int:
    """Remove manifest entries whose image files no longer exist on disk.
    Returns the number of pruned entries."""
    orphans = find_orphaned_pairs(output_dir, manifest)
    if not orphans:
        return 0
    orphan_ids = {entry.get("pair_id") for entry in orphans}
    manifest["pairs"] = [p for p in manifest.get("pairs", []) if p.get("pair_id") not in orphan_ids]
    save_manifest(output_dir, manifest)
    return len(orphan_ids)


def remove_pair(output_dir: str, manifest: dict, pair_entry: dict):
    chosen_dir = os.path.join(output_dir, "chosen")
    rejected_dir = os.path.join(output_dir, "rejected")

    for filename_key, base_dir in [("chosen_file", chosen_dir), ("rejected_file", rejected_dir)]:
        filename = pair_entry[filename_key]
        txt_name = os.path.splitext(filename)[0] + ".txt"
        for f in (filename, txt_name):
            path = find_exported_file(base_dir, f)
            if path and os.path.isfile(path):
                os.remove(path)

    pairs = manifest.get("pairs", [])
    manifest["pairs"] = [p for p in pairs if p.get("pair_id") != pair_entry.get("pair_id")]
    save_manifest(output_dir, manifest)


def finalize_export(output_dir: str, manifest: dict, val_percentage: float = 0.0) -> tuple[int, int]:
    chosen_dir = os.path.join(output_dir, "chosen")
    rejected_dir = os.path.join(output_dir, "rejected")
    chosen_train = os.path.join(chosen_dir, "train")
    chosen_val = os.path.join(chosen_dir, "val")
    rejected_train = os.path.join(rejected_dir, "train")
    rejected_val = os.path.join(rejected_dir, "val")
    for d in (chosen_train, chosen_val, rejected_train, rejected_val):
        os.makedirs(d, exist_ok=True)

    pairs = manifest.get("pairs", [])
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    val_count_target = int(len(indices) * (val_percentage / 100.0))
    val_indices = set(indices[:val_count_target])

    val_count = 0
    train_count = 0

    for i, entry in enumerate(pairs):
        is_val = i in val_indices
        target_chosen = chosen_val if is_val else chosen_train
        target_rejected = rejected_val if is_val else rejected_train

        for filename_key, base_dir, target_dir in [
            ("chosen_file", chosen_dir, target_chosen),
            ("rejected_file", rejected_dir, target_rejected),
        ]:
            src = find_exported_file(base_dir, entry[filename_key])
            dst = os.path.join(target_dir, entry[filename_key])
            if src and os.path.normpath(src) != os.path.normpath(dst):
                shutil.move(src, dst)
            txt_name = os.path.splitext(entry[filename_key])[0] + ".txt"
            txt_src = find_exported_file(base_dir, txt_name)
            txt_dst = os.path.join(target_dir, txt_name)
            if txt_src and os.path.normpath(txt_src) != os.path.normpath(txt_dst):
                shutil.move(txt_src, txt_dst)

        if is_val:
            val_count += 1
        else:
            train_count += 1

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

    return train_count, val_count


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
    multiline_captions = 0
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

        # Check caption files for multiline content
        for concept_path in (chosen_path, rejected_path):
            for root, _dirs, files in os.walk(concept_path):
                for fname in files:
                    if fname.endswith('.txt'):
                        full = os.path.join(root, fname)
                        try:
                            with open(full, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if '\n' in content.rstrip('\n'):
                                multiline_captions += 1
                        except OSError:
                            pass

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
        'multiline_captions': multiline_captions,
        'format_stats': format_stats,
        'pairs': pairs_info,
    }


def fix_multiline_captions(concept_pairs: list[tuple[str, str]]) -> int:
    """Replace newlines in caption .txt files with ', ' to make them single-line."""
    fixed = 0
    for chosen_path, rejected_path in concept_pairs:
        for concept_path in (chosen_path, rejected_path):
            for root, _dirs, files in os.walk(concept_path):
                for fname in files:
                    if not fname.endswith('.txt'):
                        continue
                    full = os.path.join(root, fname)
                    try:
                        with open(full, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except OSError:
                        continue
                    stripped = content.rstrip('\n')
                    if '\n' not in stripped:
                        continue
                    fixed_content = re.sub(r'\s*\n\s*', ', ', stripped)
                    with open(full, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    fixed += 1
    return fixed


def _copy_image(source_path: str, target_path: str):
    shutil.copy2(source_path, target_path)


def scan_finalized_pairs(concept_pairs: list[tuple[str, str]]) -> list[dict]:
    exts = supported_image_extensions()
    all_pairs: dict[str, dict] = {}

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

        all_key_set = set(chosen_keys) | set(rejected_keys)
        for key in sorted(all_key_set):
            c = chosen_keys.get(key)
            r = rejected_keys.get(key)
            all_pairs[key] = {
                'key': key,
                'chosen_path': c,
                'rejected_path': r,
                'is_orphan': c is None or r is None,
            }

    return list(all_pairs.values())


def remove_finalized_pair(chosen_path: str | None, rejected_path: str | None):
    for path in (chosen_path, rejected_path):
        if path and os.path.isfile(path):
            os.remove(path)
            txt = os.path.splitext(path)[0] + ".txt"
            if os.path.isfile(txt):
                os.remove(txt)
