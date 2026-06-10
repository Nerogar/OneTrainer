import os

from modules.util import path_util

import parse


def _normalize_relpath(relpath: str) -> str:
    return relpath.replace("\\", "/")


def pattern_has_extension(pattern: str) -> bool:
    return path_util.is_supported_image_extension(os.path.splitext(pattern)[1])


def validate_dpo_patterns(chosen_pattern: str, rejected_pattern: str):
    """Raises ValueError unless both patterns are set together and each contains
    exactly one '{}' placeholder in a relative path."""
    if bool(chosen_pattern) != bool(rejected_pattern):
        raise ValueError(
            "DPO patterns must be set together: provide both a chosen and a rejected pattern, or neither."
        )
    for label, pattern in (("chosen", chosen_pattern), ("rejected", rejected_pattern)):
        if not pattern:
            continue
        if pattern.count("{}") != 1 or "{" in pattern.replace("{}", ""):
            raise ValueError(
                f"DPO {label} pattern '{pattern}' must contain exactly one '{{}}' placeholder."
            )
        if os.path.isabs(pattern) or pattern.startswith(("/", "\\")):
            raise ValueError(
                f"DPO {label} pattern '{pattern}' must be relative to the concept path."
            )


def match_chosen(chosen_pattern: str, concept_path: str, image_path: str) -> str | None:
    """Returns the '{}' stem when image_path (inside concept_path) matches the
    chosen pattern, else None. A pattern without an image extension matches the
    extensionless relative path; one ending in an extension matches literally.
    Matching is case-insensitive and '{}' may span subdirectories."""
    relpath = _normalize_relpath(os.path.relpath(image_path, concept_path))
    if relpath == ".." or relpath.startswith("../"):
        return None
    pattern = _normalize_relpath(chosen_pattern)
    target = relpath if pattern_has_extension(pattern) else os.path.splitext(relpath)[0]
    result = parse.parse(pattern, target)
    if result is None or not result.fixed:
        return None
    return result.fixed[0]


def build_rejected_index(concept_path: str, include_subdirectories: bool) -> dict[str, list[str]]:
    """Maps lowercased extensionless relative paths to absolute file paths for
    every supported image under concept_path. Dot-directories are skipped, the
    same as mgds CollectPaths."""
    index: dict[str, list[str]] = {}

    def _walk(path: str):
        try:
            entries = [os.path.join(path, name) for name in os.listdir(path)]
        except FileNotFoundError:
            return
        for entry in entries:
            if os.path.isfile(entry):
                if path_util.is_supported_image_extension(os.path.splitext(entry)[1]):
                    relpath = _normalize_relpath(os.path.relpath(entry, concept_path))
                    key = os.path.splitext(relpath)[0].lower()
                    index.setdefault(key, []).append(entry)
            elif include_subdirectories and os.path.isdir(entry) and not os.path.basename(entry).startswith("."):
                _walk(entry)

    _walk(concept_path)
    for paths in index.values():
        paths.sort()
    return index


def resolve_rejected(
    rejected_pattern: str,
    concept_path: str,
    stem: str,
    chosen_ext: str,
    index: dict[str, list[str]],
) -> str:
    """Returns the absolute rejected path for a chosen stem. An extensionless
    pattern accepts any supported image extension, preferring the chosen
    image's own; a pattern with an extension requires it. Raises
    FileNotFoundError when no candidate exists."""
    pattern = _normalize_relpath(rejected_pattern)
    rel = pattern.replace("{}", stem)
    explicit_ext = pattern_has_extension(pattern)
    key = (os.path.splitext(rel)[0] if explicit_ext else rel).lower()
    candidates = index.get(key, [])
    if explicit_ext:
        wanted = os.path.splitext(rel)[1].lower()
        candidates = [c for c in candidates if os.path.splitext(c)[1].lower() == wanted]
    if not candidates:
        expected = rel if explicit_ext else rel + ".<image extension>"
        raise FileNotFoundError(
            f"No rejected image for '{stem}': expected '{expected}' under '{concept_path}'."
        )
    chosen_ext = chosen_ext.lower()
    return min(candidates, key=lambda c: (os.path.splitext(c)[1].lower() != chosen_ext, c))


def dpo_concept_pattern_dirs(concepts) -> list[tuple[str, str]]:
    """(chosen_dir, rejected_dir) per enabled pattern concept, for tooling that
    walks the pair folders directly (such as the RLHF tab's Check Pairs)."""
    pairs = []
    for concept in concepts:
        if not concept.enabled:
            continue
        chosen_pattern = getattr(concept, "dpo_chosen_pattern", "")
        rejected_pattern = getattr(concept, "dpo_rejected_pattern", "")
        if not chosen_pattern and not rejected_pattern:
            continue
        validate_dpo_patterns(chosen_pattern, rejected_pattern)
        chosen_dir = os.path.join(concept.path, os.path.dirname(_normalize_relpath(chosen_pattern)))
        rejected_dir = os.path.join(concept.path, os.path.dirname(_normalize_relpath(rejected_pattern)))
        pairs.append((os.path.normpath(chosen_dir), os.path.normpath(rejected_dir)))
    return pairs
