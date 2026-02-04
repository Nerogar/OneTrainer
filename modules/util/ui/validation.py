"""Input validation utilities for UI components."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod

from friendlywords import generate as generate_friendly


@dataclass(frozen=True)
class ValidationSettings:
    datetime_format: str = "%Y%m%d_%H%M%S"

SETTINGS = ValidationSettings()

ONLY_WHITESPACE = re.compile(r"^\s+$")
TRAILING_SLASH_RE = re.compile(r"[\\/]$")
ENDS_WITH_EXT = re.compile(r"\.[A-Za-z0-9]+$")
INVALID_NAMES = {"", "."}
GENERIC_MODEL_NAMES = {"lora", "embedding", "embeddings", "model", "finetune"}
TIMESTAMP_PATTERN = re.compile(r"[_-](\d{8}_\d{6}|\d{8})(?=[_.-]|$)")
# org/repo-name (alphanumeric, hyphens, underscores, dots allowed)
HUGGINGFACE_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$")

INVALID_CHARS_WIN = set('<>:"|?*')
INVALID_NAMES = {"", "."}

ValidationStatus = Literal['success', 'warning', 'error']

@dataclass
class ValidationResult:
    ok: bool
    corrected: str | None
    message: str = ""
    status: ValidationStatus | None = None

    def __post_init__(self):
        """Auto-determine status from ok field if not explicitly set."""
        if self.status is None:
            self.status = 'success' if self.ok else 'error'

    @property
    def warning(self) -> bool:
        """Backward compatibility: whether this is a warning."""
        return self.status == 'warning'

def _result(ok: bool, corrected: str | None = None, message: str = "", status: ValidationStatus | None = None) -> ValidationResult:
    return ValidationResult(ok=ok, corrected=corrected, message=message, status=status)


def _is_windows() -> bool:
    return os.name == "nt"

def _has_invalid_chars(path: str) -> bool:
    if _is_windows():
        has_drive = len(path) >= 2 and path[1] == ':' and path[0].isalpha()
        start_idx = 2 if has_drive else 0
        return any(c in INVALID_CHARS_WIN for c in path[start_idx:])
    return "\x00" in path

def _is_huggingface_repo(value: str) -> bool:
    """Check if valid HF repo after trimming whitespace"""
    trimmed = value.strip()

    if re.match(r"^https://huggingface\.[a-zA-Z]+/", trimmed):
        return True

    if trimmed.__len__() > 96:
        return False

    if " " in trimmed or "\t" in trimmed:
        return False

    if "—" in trimmed or ".." in trimmed:
        return False

    if trimmed.startswith(("\\\\", "//")):
        return False

    if trimmed.startswith("/"):
        return False

    if len(trimmed) >= 2 and trimmed[1] == ":" and trimmed[0].isalpha():
        return False

    # HF repos have exactly one slash (owner/repo)
    if trimmed.count("/") != 1:
        return False

    return bool(HUGGINGFACE_REPO_RE.match(trimmed))


def _safe_exists(path: Path, check_dir: bool = True) -> bool:
    try:
        return path.is_dir() if check_dir else path.parent.exists()
    except OSError:
        return False

def _folder_exists(path: Path) -> bool:
    return _safe_exists(path, check_dir=True)

def _parent_exists(path: Path) -> bool:
    return _safe_exists(path, check_dir=False)

def _format_path(path: Path, separator: str) -> str:
    return str(path).replace('\\', separator)

def _check_parent_exists(path: Path) -> ValidationResult | None:
    if path.parent != Path('.') and not _parent_exists(path):
        return _result(False, None, "Parent folder does not exist.")
    return None

def _has_extension(path: str | Path, extension: str) -> bool:
    return str(path).lower().endswith(extension.lower())

def _is_partial_match(current: str, required: str) -> bool:
    return len(current) > 3 and required.startswith(current)

def _has_prefix(stem: str, prefix: str) -> bool:
    """Check if stem already contains the prefix (case-insensitive)."""
    if not prefix:
        return False
    stem_lower, prefix_lower = stem.lower(), prefix.lower()
    # Exact match or prefix followed by separator
    return (stem_lower == prefix_lower or
            stem_lower.startswith((f"{prefix_lower}-", f"{prefix_lower}_")))

def _has_timestamp(stem: str) -> bool:
    return bool(TIMESTAMP_PATTERN.search(stem))

def _extension_is_substring(current_ext: str, required_ext: str) -> bool:
    current_lower = current_ext.lower()
    required_lower = required_ext.lower()
    return current_lower != required_lower and required_lower.startswith(current_lower)

def generate_default_filename(
    training_method: TrainingMethod,
    prefix: str = "",
    extension: str = ".safetensors",
    use_friendly_names: bool = False,
) -> str:
    if use_friendly_names:
        friendly = generate_friendly(1 if prefix else 2, separator="" if prefix else "-")
        name = f"{prefix}_{friendly}" if prefix else friendly
    else:
        method_str = str(training_method).lower().replace('_', '-')
        timestamp = datetime.now().strftime(SETTINGS.datetime_format)
        name = f"{prefix}_{timestamp}" if prefix else f"{method_str}_{timestamp}"
    return f"{name}{extension}"

def make_unique_filename(base_path: Path, use_friendly_names: bool = False) -> Path:
    """Generate a unique filename by appending a word or number if the path exists."""
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    if use_friendly_names:
        # Try 5 words before falling back to numbers
        for _ in range(5):
            word = generate_friendly(1, separator="")
            new_path = parent / f"{stem}_{word}{suffix}"
            if not new_path.exists():
                return new_path

    # Number-based approach (including friendly word fallback)
    counter = 1
    while counter <= 9999:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1

    return base_path

def get_allowed_formats_for_method(training_method: TrainingMethod) -> set[ModelFormat]:
    if training_method in (TrainingMethod.EMBEDDING, TrainingMethod.LORA):
        return {ModelFormat.SAFETENSORS, ModelFormat.CKPT}
    return {ModelFormat.SAFETENSORS, ModelFormat.CKPT, ModelFormat.DIFFUSERS}

@dataclass
class ValidationContext:
    raw: str
    user_input: str
    output_format: ModelFormat
    training_method: TrainingMethod
    autocorrect: bool
    prefix: str
    use_friendly_names: bool
    is_output: bool
    separator: str
    prevent_overwrite: bool
    auto_prefix: bool


def _validate_basic_input(ctx: ValidationContext) -> ValidationResult | None:
    """Perform basic input checks; return a ValidationResult to stop or None to continue."""
    if ONLY_WHITESPACE.match(ctx.raw):
        return _result(False, "", "Input cleared: empty or invalid.")

    if not ctx.user_input or ctx.user_input in INVALID_NAMES:
        return _result(False, "", "Input cleared: empty or invalid.")

    if _has_invalid_chars(ctx.user_input):
        return _result(False, None, "Input contains invalid path characters.")

    if ctx.output_format not in get_allowed_formats_for_method(ctx.training_method):
        return _result(False, None, f"{ctx.training_method} cannot output {ctx.output_format} format.")

    if not ctx.is_output:
        if not Path(ctx.user_input).exists():
            return _result(False, None, "Input path does not exist.")
        return _result(True, None, "")

    return None

def _validate_diffusers_path(ctx: ValidationContext) -> ValidationResult:
    """Validate DIFFUSERS (directory-based) paths."""
    ext_match = ENDS_WITH_EXT.search(ctx.user_input)

    if ext_match and not ctx.autocorrect:
        return _result(False, None, "DIFFUSERS expects a directory, not a file. Add a trailing slash.")

    sanitized = ctx.user_input[:ext_match.start()] if ext_match else ctx.user_input
    check_path = Path(sanitized)

    if err := _check_parent_exists(check_path):
        return err

    if not ctx.autocorrect and _folder_exists(check_path):
        msg = "Directory already exists. Stripped extension for DIFFUSERS." if ext_match else "Directory already exists."
        return _result(True, sanitized if ext_match else None, msg, 'warning')

    if ext_match:
        return _result(True, sanitized, f"Stripped extension for {ctx.output_format}.")

    return _result(True, None, "")

def _validate_single_file_path(ctx: ValidationContext) -> ValidationResult:
    """Validate single-file formats (SAFETENSORS, CKPT)."""
    required_ext = ctx.output_format.file_extension()

    is_dir_like = bool(TRAILING_SLASH_RE.search(ctx.user_input))
    path = Path(ctx.user_input)

    if not is_dir_like and '/' not in ctx.user_input and '\\' not in ctx.user_input and _folder_exists(path):
        is_dir_like = True

    if is_dir_like:
        if ctx.autocorrect:
            if not _folder_exists(path):
                return _result(False, None, "Directory does not exist.")
            default_name = generate_default_filename(ctx.training_method, ctx.prefix, required_ext, ctx.use_friendly_names)
            corrected = ctx.user_input + default_name
            return _result(True, corrected, f"Appended default filename: {default_name}")
        return _result(False, None, "Path points to a folder; a filename is required.")

    if err := _check_parent_exists(path):
        return err

    base = path.name

    if base.endswith('.'):
        if ctx.autocorrect:
            new_base = base.rstrip('.')
            new_path = _format_path(path.with_name(new_base), ctx.separator)
            return _result(True, new_path, "Removed trailing period(s).")
        return _result(False, None, "Filename cannot end with a period.")

    # Handle generic model names and auto-prefix
    stem_lower = path.stem.lower()

    # Check for generic names that should be prefixed
    if stem_lower in GENERIC_MODEL_NAMES and ctx.prefix and ctx.autocorrect:
        new_stem = f"{ctx.prefix}_{path.stem}"
        new_path = _format_path(path.with_name(new_stem + path.suffix), ctx.separator)
        return _result(True, new_path, f"Added prefix to generic name '{path.stem}'.", 'warning')
    elif stem_lower in GENERIC_MODEL_NAMES and ctx.prefix:
        return _result(True, None, f"WARNING: Generic filename '{path.stem}' may cause confusion. Consider adding a prefix.", 'warning')

    # Auto-prefix: add prefix to manually-entered filenames
    if ctx.auto_prefix and ctx.prefix and ctx.autocorrect and not _has_prefix(path.stem, ctx.prefix):
        new_stem = f"{ctx.prefix}-{path.stem}"
        new_path = _format_path(path.with_name(new_stem + path.suffix), ctx.separator)
        return _result(True, new_path, f"Added prefix '{ctx.prefix}' to filename.", 'warning')

    if _has_extension(base, required_ext):
        # Check if file exists and handle overwrite prevention
        if path.exists():
            if ctx.prevent_overwrite and ctx.autocorrect:
                # Generate a unique filename
                unique_path = make_unique_filename(path, ctx.use_friendly_names)
                if unique_path != path:
                    corrected = _format_path(unique_path, ctx.separator)
                    suffix_added = unique_path.stem[len(path.stem):]  # Get what was added
                    return _result(True, corrected, f"File exists. Added '{suffix_added}' to prevent overwrite.")
            return _result(True, None, "WARNING: File already exists and will be overwritten.", 'warning')
        return _result(True, None, "")

    ext_match = ENDS_WITH_EXT.search(base)
    if ext_match:
        current_ext = ext_match.group(0).lower()
        chars_after_dot = len(current_ext) - 1
        start_idx = ext_match.start()

        # Check if current extension is a partial match of required (e.g., .safetensor vs .safetensors)
        if _extension_is_substring(current_ext, required_ext):
            if ctx.autocorrect:
                new_base = base[:start_idx] + required_ext
                new_path = _format_path(path.with_name(new_base), ctx.separator)
                return _result(True, new_path, f"Completed {current_ext} -> {required_ext}")
            return _result(False, None, f"Extension is incomplete; expected {required_ext}")

        if chars_after_dot >= 3:
            if _is_partial_match(current_ext, required_ext):
                if ctx.autocorrect:
                    new_base = base[:start_idx] + required_ext
                    new_path = _format_path(path.with_name(new_base), ctx.separator)
                    return _result(True, new_path, f"Completed {current_ext} -> {required_ext}")
                return _result(False, None, f"Extension is incomplete; expected {required_ext}")
            else:
                if ctx.autocorrect:
                    new_base = base[:start_idx] + required_ext
                    new_path = _format_path(path.with_name(new_base), ctx.separator)
                    return _result(True, new_path, f"Replaced {current_ext} with {required_ext}")
                return _result(False, None, f"Wrong extension; expected {required_ext}")

        if 1 <= chars_after_dot < 3 and required_ext.startswith(current_ext):
            return _result(False, None, f"Extension {current_ext} is too short; Type at least 3 characters for auto-correction to occur.")

        if ctx.autocorrect:
            # Replace unrecognized extension with required
            new_base = base[:start_idx] + required_ext
            new_path = _format_path(path.with_name(new_base), ctx.separator)
            return _result(True, new_path, f"Replaced {current_ext} with {required_ext}")
        return _result(False, None, f"Filename must end with {required_ext}")

    # No extension - append required extension
    if ctx.autocorrect:
        corrected = _format_path(Path(path.parent, base + required_ext), ctx.separator)
        return _result(True, corrected, f"Appended {required_ext} extension.")
    return _result(False, None, f"Filename must end with {required_ext}")

def validate_destination(
    raw: str,
    output_format: ModelFormat,
    training_method: TrainingMethod,
    autocorrect: bool = True,
    prefix: str = "",
    use_friendly_names: bool = False,
    is_output: bool = True,
    prevent_overwrite: bool = False,
    auto_prefix: bool = False,
    skip_overwrite_protection: bool = False,
) -> ValidationResult:
    """
    Validate and optionally auto-correct a model output destination path.

    This function handles:
    - Empty/invalid input
    - Format/method compatibility
    - Directory vs file paths
    - File extension validation and correction
    - Parent directory existence
    - Existing file warnings
    - Automatic overwrite prevention

    Args:
        raw: Raw user input string
        output_format: Target output format (DIFFUSERS, SAFETENSORS, CKPT)
        training_method: Training method being used
        autocorrect: Whether to auto-correct invalid input
        prefix: Optional prefix for auto-generated filenames
        use_friendly_names: Use friendly names instead of timestamps
        is_output: Whether this is an output path (True) or input path (False)
        prevent_overwrite: Automatically make filename unique if it would overwrite
        auto_prefix: Automatically add prefix to manually-entered filenames
        skip_overwrite_protection: Skip overwrite protection (used when validation triggered by settings changes)

    Returns:
        ValidationResult with ok status, optional corrected value, and message
    """
    ctx = ValidationContext(
        raw=raw,
        user_input=raw.strip(),
        output_format=output_format,
        training_method=training_method,
        autocorrect=autocorrect,
        prefix=prefix,
        use_friendly_names=use_friendly_names,
        is_output=is_output,
        separator='/' if '/' in raw else '\\',
        prevent_overwrite=prevent_overwrite and not skip_overwrite_protection,
        auto_prefix=auto_prefix,
    )

    if basic_check := _validate_basic_input(ctx):
        return basic_check

    if output_format == ModelFormat.DIFFUSERS:
        return _validate_diffusers_path(ctx)
    else:
        return _validate_single_file_path(ctx)

def validate_basic_type(
    value: str,
    declared_type: type,
    nullable: bool,
    default_val: Any,
) -> ValidationResult:
    """Validate basic Python types (int, float, bool, str)."""
    # Handle empty values
    if value == "":
        if nullable or (declared_type is str and default_val == ""):
            return _result(True, None, "")
        return _result(False, None, "Value required")

    # Type validation
    try:
        if declared_type is int:
            int(value)
        elif declared_type is float:
            float(value)
        elif declared_type is bool:
            if value.lower() not in ("true", "false", "0", "1"):
                return _result(False, None, "Invalid bool")
        return _result(True, None, "")
    except ValueError:
        type_name = declared_type.__name__ if hasattr(declared_type, '__name__') else str(declared_type)
        return _result(False, None, f"Invalid {type_name}")

def validate_file_path(
    value: str,
    is_output: bool = False,
    valid_extensions: list[str] | None = None,
    path_type: str = "file",
) -> ValidationResult:
    """Validate a file or directory path."""
    if not value:
        return _result(True, None, "")

    # For input paths (not output), accept HuggingFace repo format
    if not is_output and _is_huggingface_repo(value):
        return _result(True, None, "")

    # Check for invalid characters
    if _has_invalid_chars(value):
        return _result(False, None, "Input contains invalid path characters.")

    path = Path(value)

    if path_type == "directory":
        if is_output:
            if err := _check_parent_exists(path):
                return err
        else:
            if not _folder_exists(path):
                return _result(False, None, "Directory does not exist.")
    else:
        # For files
        if valid_extensions:
            if not any(_has_extension(value, ext) for ext in valid_extensions):
                return _result(False, None, f"File must have one of these extensions: {', '.join(valid_extensions)}")

        if is_output:
            if err := _check_parent_exists(path):
                return err
        else:
            if not path.exists():
                return _result(False, None, "File does not exist")

    return _result(True, None, "")
