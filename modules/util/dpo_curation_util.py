import hashlib
import json
import os
import random
import re
import shutil
import threading

_SHA256_CHUNK_BYTES = 65536

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.image_metadata_util import strip_angle_bracket_segments
from modules.util.path_util import supported_image_extensions

UNCONDITIONAL_PROMPT = "UNCONDITIONAL"


def walk_skipping_dotted(folder: str):
    """``os.walk`` wrapper that prunes dot-prefixed subdirectories in-place
    (``.thumbnails``, ``.cache``, ``.stversions``, ...) so their contents are
    never yielded — gallery apps and sync tools plant resized previews there
    that would otherwise dhash-collide with (or duplicate) the real images.
    Yields ``(root, files)`` pairs. The top-level ``folder`` itself is always
    walked, even when its own name starts with a dot — only subdirectories
    are filtered."""
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        yield root, files


# Mirrors mgds AspectBucketing.all_possible_input_aspects: the trainer snaps
# every image's aspect to the nearest of these ratios (and their inverses)
# regardless of target resolution, then center-crops to fit. Each entry is the
# (W, H) label parts of the landscape orientation — aspect value W/H.
_TRAINER_BUCKET_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (5, 4),
    (3, 2),
    (7, 4),
    (2, 1),
    (5, 2),
    (3, 1),
    (7, 2),
    (4, 1),
)


def trainer_bucket_key(width: float, height: float) -> str:
    """Label (``7:4``, ``4:7``, ``1:1``, ...) of the trainer aspect bucket the
    image falls into, mirroring AspectBucketing's nearest-aspect assignment
    (argmin over ``h/w`` against every bucket aspect and its inverse). Returns
    ``""`` for non-positive inputs so the caller can fall through to its own
    handling.

    Bucket-level grouping is what lets 1344x768 and 1680x960 (both ~16:9
    inference presets, true ratio 1.75) curate together: the trainer crops
    both to the same bucket, so requiring exact pixel-ratio equality would
    split pairs the trainer itself considers identical."""
    if width <= 0 or height <= 0:
        return ""
    aspect = height / width
    best_label = ""
    best_diff: float | None = None
    for num, den in _TRAINER_BUCKET_RATIOS:
        for bucket_aspect, label in ((den / num, f"{num}:{den}"), (num / den, f"{den}:{num}")):
            diff = abs(bucket_aspect - aspect)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_label = label
    return best_label


def _parse_aspect_string(value: str) -> tuple[float, float] | None:
    """Parse a ``16:9`` / ``16x9`` / ``1.78`` style aspect string into a
    (width, height) ratio pair, or None when it doesn't look like one."""
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:/xX×]\s*(\d+(?:\.\d+)?)", cleaned)
    if match:
        w, h = float(match.group(1)), float(match.group(2))
        return (w, h) if w > 0 and h > 0 else None
    try:
        ratio = float(cleaned)
    except ValueError:
        return None
    return (ratio, 1.0) if ratio > 0 else None


def normalize_aspect_for_grouping(aspectratio: str) -> str:
    """Map any stored/metadata aspect string onto its trainer bucket label so
    manifest entries written before bucket-aware grouping (SwarmUI ``16:9``,
    exact reductions like ``1343:768``) count against the same groups as
    freshly scanned images. Strings that don't parse pass through unchanged."""
    parsed = _parse_aspect_string(aspectratio)
    if parsed is None:
        return (aspectratio or "").strip()
    return trainer_bucket_key(*parsed)


def resolve_aspect_ratio(meta_aspectratio: str, image_path: str) -> str:
    """Trainer bucket label for the image, preferring actual pixel dimensions —
    AspectBucketing never reads generator metadata, so pixels are the
    authority (a SwarmUI ``16:9`` preset actually emits a 1.75 image). Falls
    back to parsing the metadata string when the image can't be decoded, and
    ``""`` when neither works. PIL ``Image.open`` is lazy and only reads the
    header, so the per-file cost is one small read."""
    width = height = 0
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        pass
    if width > 0 and height > 0:
        return trainer_bucket_key(width, height)
    return normalize_aspect_for_grouping(meta_aspectratio)


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


def compute_file_sha256(path: str) -> str | None:
    """Streaming SHA-256 of a file's bytes — returns hex digest or None on I/O error.
    Used to detect byte-identical images across renames or re-saved copies that
    path-based filtering can't catch."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_SHA256_CHUNK_BYTES)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def compute_pixel_hash(path: str) -> str | None:
    """BLAKE3 over the decoded pixel buffer (mode, size, raw bytes) rather than
    the file: PNG text chunks, EXIF, ICC profiles, and re-encoded container
    metadata don't count as a difference, but a single different pixel does.
    Falls back to BLAKE3 of the raw file bytes when PIL can't decode (corrupt,
    unsupported, zero-byte) so the file still gets a stable hash. Returns None
    only when the file can't be read at all."""
    import blake3
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            im.load()
            hasher = blake3.blake3()
            hasher.update(f"{im.mode}|{im.size[0]}x{im.size[1]}|".encode())
            hasher.update(im.tobytes())
            return hasher.hexdigest()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        pass
    try:
        hasher = blake3.blake3()
        hasher.update_mmap(path)
        return hasher.hexdigest()
    except (OSError, ValueError):
        pass
    try:
        hasher = blake3.blake3()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


class DpoScanCache:
    """Thread-safe, JSON-backed per-file cache for DPO curation scans, keyed by
    absolute path and validated by (size, mtime_ns). Caches the two expensive
    per-file operations: the pixel-content hash (a full image decode) and the
    grouping key (metadata extraction — which, for files without embedded
    prompt markers, reads the whole file and scans it character by character).
    With both cached, rescanning an unchanged folder costs one ``os.stat`` per
    file. Field getters are safe to call from a thread pool — the expensive
    compute happens outside the lock so workers don't serialize on each other.

    Bump ``_VERSION`` whenever the semantics of a cached field change (hash
    algorithm, prompt normalization, aspect-ratio derivation): stale values
    are otherwise served forever for unchanged files."""

    # v2: group_key aspect component changed from exact reduced/metadata
    # strings to trainer bucket labels (trainer_bucket_key).
    _VERSION = 2

    def __init__(self, cache_path: str):
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._dirty = False
        # Rough counters for UI feedback ("N served from cache"). Updated from
        # worker threads without atomics — close enough for a progress display.
        self.hits = 0
        self.misses = 0
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and data.get("version") == self._VERSION
                and isinstance(data.get("entries"), dict)
            ):
                self._entries = data["entries"]
        except (OSError, json.JSONDecodeError):
            pass

    def _get_field(self, path: str, field: str, compute):
        """Cached ``field`` for ``path``, recomputing via ``compute(path)`` on a
        size/mtime mismatch or when the field is missing. Fields merge into one
        entry per file, so computing the hash never discards a cached group key
        (and vice versa)."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = os.path.normcase(os.path.abspath(path))
        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.get("size") == st.st_size
                and entry.get("mtime_ns") == st.st_mtime_ns
                and field in entry
            ):
                self.hits += 1
                return entry[field]
        self.misses += 1
        value = compute(path)
        if value is None:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.get("size") != st.st_size or entry.get("mtime_ns") != st.st_mtime_ns:
                entry = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                self._entries[key] = entry
            entry[field] = value
            self._dirty = True
        return value

    def get_pixel_hash(self, path: str) -> str | None:
        return self._get_field(path, "hash", compute_pixel_hash)

    def get_group_key(self, path: str, compute) -> tuple[str, str] | None:
        """Cached (prompt, aspectratio) grouping key. ``compute(path)`` must
        return that tuple; exceptions propagate so callers can count the file
        as scanned-but-unusable, matching uncached behavior."""
        value = self._get_field(path, "group_key", lambda p: list(compute(p)))
        if value is None:
            return None
        return tuple(value)

    def save(self) -> None:
        """Persist to disk (atomic replace). Entries whose file no longer exists
        are pruned so renames/finalize moves don't grow the cache forever. No-op
        when nothing changed since the last save. The lock is held across the
        file write so concurrent saves (e.g. cancel_session racing the scan
        worker's final save) serialize instead of corrupting the file."""
        with self._lock:
            if not self._dirty:
                return
            self._entries = {key: entry for key, entry in self._entries.items() if os.path.isfile(key)}
            self._dirty = False
            tmp_path = self._cache_path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump({"version": self._VERSION, "entries": self._entries}, f)
                os.replace(tmp_path, self._cache_path)
            except OSError:
                pass


def _backfill_pair_fingerprint(entry: dict, output_dir: str) -> bool:
    """Populate missing chosen/rejected sha256+size on an old manifest `entry` by
    hashing the original source path or the exported copy. Returns True if any
    field was added. Existing fingerprints are left alone."""
    modified = False
    sides = (
        ("chosen_source", "chosen_sha256", "chosen_size", "chosen_file", "chosen"),
        ("rejected_source", "rejected_sha256", "rejected_size", "rejected_file", "rejected"),
    )
    for source_key, hash_key, size_key, file_key, subdir in sides:
        if entry.get(hash_key) and isinstance(entry.get(size_key), int):
            continue
        candidate_paths: list[str] = []
        source = entry.get(source_key)
        if source and os.path.isfile(source):
            candidate_paths.append(source)
        exported = find_exported_file(os.path.join(output_dir, subdir), entry.get(file_key, ""))
        if exported and exported not in candidate_paths:
            candidate_paths.append(exported)
        for path in candidate_paths:
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            digest = compute_file_sha256(path)
            if digest is None:
                continue
            entry[hash_key] = digest
            entry[size_key] = size
            modified = True
            break
    return modified


def manifest_used_fingerprints(manifest: dict, output_dir: str | None = None) -> dict[int, set[str]]:
    """Map of ``{file_size: {sha256_hex, ...}}`` for every pair in the manifest. When
    ``output_dir`` is provided, entries missing fingerprints are backfilled from the
    source path or exported copy and the manifest is saved back so the cost is paid
    only once. Designed for ``is_byte_identical_used`` — the size key acts as a
    cheap pre-filter so we only hash candidates whose size matches an existing
    pair."""
    fingerprints: dict[int, set[str]] = {}
    modified = False
    for entry in manifest.get("pairs", []):
        if output_dir is not None and _backfill_pair_fingerprint(entry, output_dir):
            modified = True
        for hash_key, size_key in (("chosen_sha256", "chosen_size"), ("rejected_sha256", "rejected_size")):
            digest = entry.get(hash_key)
            size = entry.get(size_key)
            if digest and isinstance(size, int):
                fingerprints.setdefault(size, set()).add(digest)
    if modified and output_dir is not None:
        save_manifest(output_dir, manifest)
    return fingerprints


def is_byte_identical_used(used_fingerprints: dict[int, set[str]], path: str) -> bool:
    """True when ``path``'s bytes match any pair in ``used_fingerprints``. Skips
    hashing when no committed pair shares the candidate's size, so an empty/small
    manifest makes this nearly free."""
    if not used_fingerprints:
        return False
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    candidates = used_fingerprints.get(size)
    if not candidates:
        return False
    digest = compute_file_sha256(path)
    return digest is not None and digest in candidates


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
        # Aspect strings are normalized to trainer bucket labels so pairs
        # exported before bucket-aware grouping ("16:9", exact reductions)
        # count against the same group as newly scanned images.
        key = (
            normalize_prompt_for_grouping(entry["prompt"]),
            normalize_aspect_for_grouping(entry.get("aspectratio", "")),
        )
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

    # Capture fingerprints before copy — keeps the dedup check honest even if the
    # exported copy is later edited in place or replaced.
    try:
        chosen_size: int | None = os.path.getsize(chosen_path)
    except OSError:
        chosen_size = None
    try:
        rejected_size: int | None = os.path.getsize(rejected_path)
    except OSError:
        rejected_size = None
    chosen_sha256 = compute_file_sha256(chosen_path)
    rejected_sha256 = compute_file_sha256(rejected_path)

    _copy_image(chosen_path, os.path.join(chosen_dir, safe_name + chosen_ext))
    _copy_image(rejected_path, os.path.join(rejected_dir, safe_name + rejected_ext))
    # Single-concept pattern pairing reads the caption from the chosen side only.
    with open(os.path.join(chosen_dir, safe_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(caption)

    manifest.setdefault("pairs", []).append(
        {
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
            # SHA-256 + size catch byte-identical images even after the user
            # renames, moves, or duplicates them — path matching alone misses
            # those.
            "chosen_sha256": chosen_sha256,
            "rejected_sha256": rejected_sha256,
            "chosen_size": chosen_size,
            "rejected_size": rejected_size,
        }
    )
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

    _write_pattern_concepts(os.path.abspath(output_dir))

    return train_count, val_count


def _write_pattern_concepts(abs_output: str):
    """Writes a concepts.json with one pattern concept per train/val split. The
    concept path is the export root; the patterns pair chosen/<split>/<stem>
    with rejected/<split>/<stem> inside one sample."""
    concepts = []
    for subdir, concept_type in (("train", ConceptType.STANDARD), ("val", ConceptType.VALIDATION)):
        cfg = ConceptConfig.default_values()
        cfg.name = f"DPO {subdir}"
        cfg.path = abs_output
        cfg.type = concept_type
        cfg.enabled = True
        cfg.include_subdirectories = True
        cfg.dpo_chosen_pattern = f"chosen/{subdir}/{{}}"
        cfg.dpo_rejected_pattern = f"rejected/{subdir}/{{}}"
        concepts.append(cfg.to_dict())
    with open(os.path.join(abs_output, "concepts.json"), "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)


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
            with open(os.path.join(chosen_dir, safe_name + ".txt"), "w", encoding="utf-8") as f:
                f.write(caption)

        if is_val:
            val_count += len(pairs)
        else:
            train_count += len(pairs)

    _write_pattern_concepts(os.path.abspath(output_dir))

    return (
        chosen_train_dir,
        rejected_train_dir,
        chosen_val_dir,
        rejected_val_dir,
        skipped_count,
        val_count,
        train_count,
    )


def dpo_pair_key(image_path: str, concept_path: str) -> str:
    try:
        relative = os.path.relpath(image_path, concept_path)
    except ValueError:
        relative = os.path.basename(image_path)
    return os.path.splitext(relative.replace("\\", "/"))[0]


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
                ext = os.path.splitext(img_path)[1].lower().lstrip(".")
                format_stats[ext] = format_stats.get(ext, 0) + 1

        # Check caption files for multiline content
        for concept_path in (chosen_path, rejected_path):
            for root, _dirs, files in os.walk(concept_path):
                for fname in files:
                    if fname.endswith(".txt"):
                        full = os.path.join(root, fname)
                        try:
                            with open(full, "r", encoding="utf-8") as f:
                                content = f.read()
                            if "\n" in content.rstrip("\n"):
                                multiline_captions += 1
                        except OSError:
                            pass

        all_matched += len(matched_keys)
        all_chosen_stray += chosen_stray
        all_rejected_stray += rejected_stray
        pairs_info.append(
            {
                "chosen_path": chosen_path,
                "rejected_path": rejected_path,
                "matched": len(matched_keys),
                "chosen_stray": chosen_stray,
                "rejected_stray": rejected_stray,
            }
        )

    return {
        "total_matched": all_matched,
        "total_chosen_stray": all_chosen_stray,
        "total_rejected_stray": all_rejected_stray,
        "multiline_captions": multiline_captions,
        "format_stats": format_stats,
        "pairs": pairs_info,
    }


def fix_multiline_captions(concept_pairs: list[tuple[str, str]]) -> int:
    """Replace newlines in caption .txt files with ', ' to make them single-line."""
    fixed = 0
    for chosen_path, rejected_path in concept_pairs:
        for concept_path in (chosen_path, rejected_path):
            for root, _dirs, files in os.walk(concept_path):
                for fname in files:
                    if not fname.endswith(".txt"):
                        continue
                    full = os.path.join(root, fname)
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            content = f.read()
                    except OSError:
                        continue
                    stripped = content.rstrip("\n")
                    if "\n" not in stripped:
                        continue
                    fixed_content = re.sub(r"\s*\n\s*", ", ", stripped)
                    with open(full, "w", encoding="utf-8") as f:
                        f.write(fixed_content)
                    fixed += 1
    return fixed


def _read_caption(image_path: str) -> tuple[str | None, str]:
    """Return (txt_path, caption_text) for the .txt sidecar of `image_path`.
    txt_path is None if no sidecar exists; caption_text is '' in that case."""
    txt = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.isfile(txt):
        return None, ""
    try:
        with open(txt, "r", encoding="utf-8") as f:
            return txt, f.read()
    except OSError:
        return txt, ""


def find_caption_mismatches(concept_pairs: list[tuple[str, str]]) -> list[dict]:
    """For each chosen/rejected image pair that matches by basename key, compare
    the .txt sidecar contents. Return entries where the captions differ.

    Compares with .rstrip('\\n') on both sides so a trailing-newline difference
    does not register as a mismatch (use fix_multiline_captions for that).
    A missing sidecar on one side counts as a mismatch with caption == ''.
    """
    exts = supported_image_extensions()
    mismatches: list[dict] = []

    for index, (chosen_path, rejected_path) in enumerate(concept_pairs):
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

        for key in sorted(set(chosen_keys) & set(rejected_keys)):
            chosen_image = chosen_keys[key]
            rejected_image = rejected_keys[key]
            chosen_txt, chosen_caption = _read_caption(chosen_image)
            rejected_txt, rejected_caption = _read_caption(rejected_image)
            captions_match = chosen_caption.rstrip("\n") == rejected_caption.rstrip("\n")
            both_present = chosen_txt is not None and rejected_txt is not None
            if captions_match and both_present:
                continue
            mismatches.append(
                {
                    "concept_pair_index": index,
                    "key": key,
                    "chosen_image": chosen_image,
                    "rejected_image": rejected_image,
                    "chosen_caption_path": chosen_txt,
                    "rejected_caption_path": rejected_txt,
                    "chosen_caption": chosen_caption,
                    "rejected_caption": rejected_caption,
                }
            )

    return mismatches


def apply_caption_to_pair(chosen_image: str, rejected_image: str, caption_text: str) -> None:
    """Write `caption_text` to BOTH .txt sidecars for the pair. Creates the file
    if missing, overwrites if it exists. UTF-8, no trailing newline added."""
    for image_path in (chosen_image, rejected_image):
        if not image_path:
            continue
        txt = os.path.splitext(image_path)[0] + ".txt"
        with open(txt, "w", encoding="utf-8") as f:
            f.write(caption_text)


def correct_all_captions_to_chosen(mismatches: list[dict]) -> int:
    """For each mismatch entry, overwrite the rejected .txt with the chosen
    caption text. Returns the number of pairs corrected."""
    corrected = 0
    for entry in mismatches:
        rejected_image = entry.get("rejected_image")
        if not rejected_image:
            continue
        caption_text = entry.get("chosen_caption", "")
        txt = os.path.splitext(rejected_image)[0] + ".txt"
        with open(txt, "w", encoding="utf-8") as f:
            f.write(caption_text)
        corrected += 1
    return corrected


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
                "key": key,
                "chosen_path": c,
                "rejected_path": r,
                "is_orphan": c is None or r is None,
            }

    return list(all_pairs.values())


def remove_finalized_pair(chosen_path: str | None, rejected_path: str | None):
    for path in (chosen_path, rejected_path):
        if path and os.path.isfile(path):
            os.remove(path)
            txt = os.path.splitext(path)[0] + ".txt"
            if os.path.isfile(txt):
                os.remove(txt)
