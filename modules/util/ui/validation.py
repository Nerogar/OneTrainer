"""
Input validation utilities for OneTrainer UI components.

Provides validation for file paths, directories, and model output destinations
with support for auto-correction and detailed error messaging.
"""

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

# Regex patterns
ONLY_WHITESPACE = re.compile(r"^\s+$")
TRAILING_SLASH_RE = re.compile(r"[\\/]$")
ENDS_WITH_EXT = re.compile(r"\.[A-Za-z0-9]+$")

# Invalid characters/names
INVALID_CHARS_WIN = set('<>:"|?*')
INVALID_NAMES = {"", "."}

ValidationStatus = Literal['success', 'warning', 'error']

@dataclass
class ValidationResult:
    """Result of a validation operation."""
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

# Helper to create common results
def _result(ok: bool, corrected: str | None = None, message: str = "", status: ValidationStatus | None = None) -> ValidationResult:
    return ValidationResult(ok=ok, corrected=corrected, message=message, status=status)


def _is_windows() -> bool:
    return os.name == "nt"

def _has_invalid_chars(path: str) -> bool:
    """Check if path contains invalid characters."""
    if _is_windows():
        has_drive = len(path) >= 2 and path[1] == ':' and path[0].isalpha()
        start_idx = 2 if has_drive else 0
        return any(c in INVALID_CHARS_WIN for c in path[start_idx:])
    return "\x00" in path

def _safe_exists(path: Path, check_dir: bool = True) -> bool:
    """Safely check if a path exists."""
    try:
        return path.is_dir() if check_dir else path.parent.exists()
    except OSError:
        return False

def _folder_exists(path: Path) -> bool:
    return _safe_exists(path, check_dir=True)

def _parent_exists(path: Path) -> bool:
    return _safe_exists(path, check_dir=False)

def _format_path(path: Path, separator: str) -> str:
    """Format path with preferred separator."""
    return str(path).replace('\\', separator)

def _has_extension(path: str | Path, extension: str) -> bool:
    """Check if path has the given extension (case-insensitive)."""
    return str(path).lower().endswith(extension.lower())

def _get_extension(path: str | Path) -> str | None:
    """Extract extension from path, or None if no extension."""
    match = ENDS_WITH_EXT.search(str(path))
    return match.group(0).lower() if match else None

def _is_partial_match(current: str, required: str) -> bool:
    """Check if current extension is a partial match of required."""
    chars_after_dot = len(current) - 1
    return chars_after_dot >= 3 and required.startswith(current)

def generate_default_filename(
    training_method: TrainingMethod,
    prefix: str = "",
    extension: str = ".safetensors",
    use_friendly_names: bool = False,
) -> str:
    """Generate a default filename for model output."""
    if use_friendly_names:
        friendly = generate_friendly(1 if prefix else 2, separator="" if prefix else "-")
        name = f"{prefix}_{friendly}" if prefix else friendly
    else:
        method_str = str(training_method).lower().replace('_', '-')
        timestamp = datetime.now().strftime(SETTINGS.datetime_format)
        name = f"{prefix}_{timestamp}" if prefix else f"{method_str}_{timestamp}"
    return f"{name}{extension}"

def get_allowed_formats_for_method(training_method: TrainingMethod) -> set[ModelFormat]:
    """Get the allowed output formats for a given training method."""
    if training_method in (TrainingMethod.EMBEDDING, TrainingMethod.LORA):
        return {ModelFormat.SAFETENSORS, ModelFormat.CKPT}
    return {ModelFormat.SAFETENSORS, ModelFormat.CKPT, ModelFormat.DIFFUSERS}

@dataclass
class ValidationContext:
    """Context for validation operations."""
    raw: str
    user_input: str
    output_format: ModelFormat
    training_method: TrainingMethod
    autocorrect: bool
    prefix: str
    use_friendly_names: bool
    is_output: bool
    separator: str


def _validate_basic_input(ctx: ValidationContext) -> ValidationResult | None:
    """Perform basic validation checks on input. Returns None to continue."""

    # Check for whitespace-only input
    if ONLY_WHITESPACE.match(ctx.raw):
        return _result(False, "", "Input cleared: empty or invalid.")

    # Check for empty or invalid names
    if not ctx.user_input or ctx.user_input in INVALID_NAMES:
        return _result(False, "", "Input cleared: empty or invalid.")

    # Check for invalid characters
    if _has_invalid_chars(ctx.user_input):
        return _result(False, None, "Input contains invalid path characters.")

    # Check format/method compatibility
    if ctx.output_format not in get_allowed_formats_for_method(ctx.training_method):
        return _result(False, None, f"{ctx.training_method} cannot output {ctx.output_format} format.")

    # For input paths, verify existence
    if not ctx.is_output:
        if not Path(ctx.user_input).exists():
            return _result(False, None, "Input path does not exist.")
        return _result(True, None, "")

    return None

def _validate_diffusers_path(ctx: ValidationContext) -> ValidationResult:
    """Validate DIFFUSERS format path (directory-based)."""
    ext_match = ENDS_WITH_EXT.search(ctx.user_input)

    if ext_match and not ctx.autocorrect:
        return _result(False, None, "DIFFUSERS expects a directory, not a file. Add a trailing slash.")

    # Remove extension if present
    sanitized = ctx.user_input[:ext_match.start()] if ext_match else ctx.user_input
    check_path = Path(sanitized)

    # Check parent directory exists
    if check_path.parent != Path('.') and not _parent_exists(check_path):
        return _result(False, None, "Parent folder does not exist.")

    # Warn if directory already exists
    if not ctx.autocorrect and _folder_exists(check_path):
        msg = "Directory already exists. Stripped extension for DIFFUSERS." if ext_match else "Directory already exists."
        return _result(True, sanitized if ext_match else None, msg, 'warning')

    # Extension was stripped
    if ext_match:
        return _result(True, sanitized, f"Stripped extension for {ctx.output_format}.")

    return _result(True, None, "")

def _validate_single_file_path(ctx: ValidationContext) -> ValidationResult:
    """Validate single-file format path (SAFETENSORS, CKPT)."""
    required_ext = ctx.output_format.file_extension()

    # Check if input looks like a directory
    is_dir_like = bool(TRAILING_SLASH_RE.search(ctx.user_input))
    path = Path(ctx.user_input)

    # If no slash but exists as folder, treat as directory
    if not is_dir_like and '/' not in ctx.user_input and '\\' not in ctx.user_input and _folder_exists(path):
        is_dir_like = True

    # Handle directory-like input
    if is_dir_like:
        if ctx.autocorrect:
            if not _folder_exists(path):
                return _result(False, None, "Directory does not exist.")
            default_name = generate_default_filename(ctx.training_method, ctx.prefix, required_ext, ctx.use_friendly_names)
            corrected = ctx.user_input + default_name
            return _result(True, corrected, f"Appended default filename: {default_name}")
        return _result(False, None, "Path points to a folder; a filename is required.")

    # Check parent directory exists
    if path.parent != Path('.') and not _parent_exists(path):
        return _result(False, None, "Parent folder does not exist.")

    base = path.name

    # Handle trailing periods
    if base.endswith('.'):
        if ctx.autocorrect:
            new_base = base.rstrip('.')
            new_path = _format_path(path.with_name(new_base), ctx.separator)
            return _result(True, new_path, "Removed trailing period(s).")
        return _result(False, None, "Filename cannot end with a period.")

    # Check if already has correct extension
    if _has_extension(base, required_ext):
        if path.exists():
            return _result(True, None, "WARNING: File already exists and will be overwritten.", 'warning')
        return _result(True, None, "")

    # Check for partial or incorrect extension
    ext_match = ENDS_WITH_EXT.search(base)
    if ext_match:
        current_ext = ext_match.group(0).lower()
        chars_after_dot = len(current_ext) - 1
        start_idx = ext_match.start()

        # Extension has at least 3 characters after dot
        if chars_after_dot >= 3:
            # Check if it's a partial match (e.g., ".safe" for ".safetensors")
            if _is_partial_match(current_ext, required_ext):
                if ctx.autocorrect:
                    new_base = base[:start_idx] + required_ext
                    new_path = _format_path(path.with_name(new_base), ctx.separator)
                    return _result(True, new_path, f"Completed {current_ext} -> {required_ext}")
                return _result(False, None, f"Extension is incomplete; expected {required_ext}")
            else:
                # Different extension entirely
                if ctx.autocorrect:
                    new_base = base[:start_idx] + required_ext
                    new_path = _format_path(path.with_name(new_base), ctx.separator)
                    return _result(True, new_path, f"Replaced {current_ext} with {required_ext}")
                return _result(False, None, f"Wrong extension; expected {required_ext}")

        # Extension too short for auto-correction
        if 1 <= chars_after_dot < 3 and required_ext.startswith(current_ext):
            return _result(False, None, f"Extension {current_ext} is too short; Type at least 3 characters for auto-correction to occur.")

        # Unrecognized short extension - append required
        if ctx.autocorrect:
            corrected = _format_path(Path(path.parent, base + required_ext), ctx.separator)
            return _result(True, corrected, f"Appended {required_ext} extension.")
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

    Args:
        raw: Raw user input string
        output_format: Target output format (DIFFUSERS, SAFETENSORS, CKPT)
        training_method: Training method being used
        autocorrect: Whether to auto-correct invalid input
        prefix: Optional prefix for auto-generated filenames
        use_friendly_names: Use friendly names instead of timestamps
        is_output: Whether this is an output path (True) or input path (False)

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
        separator='/' if '/' in raw else '\\'
    )

    # Basic validation checks
    if basic_check := _validate_basic_input(ctx):
        return basic_check

    # Format-specific validation
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
    """
    Validate basic Python types (int, float, bool, str).

    Args:
        value: The string value to validate
        declared_type: The expected Python type (int, float, bool, str)
        nullable: Whether None/empty is allowed
        default_val: The default value for this field

    Returns:
        ValidationResult with validation outcome
    """
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
    """
    Validate a file or directory path.

    Args:
        value: Path string to validate
        is_output: Whether this is an output path
        valid_extensions: List of valid extensions (with dots), or None for any
        path_type: Either "file" or "directory"

    Returns:
        ValidationResult with validation outcome
    """
    if not value:
        return _result(True, None, "")

    # Check for invalid characters
    if _has_invalid_chars(value):
        return _result(False, None, "Input contains invalid path characters.")

    path = Path(value)

    if path_type == "directory":
        if is_output:
            if path.parent != Path('.') and not _parent_exists(path):
                return _result(False, None, "Parent folder does not exist.")
        else:
            if not _folder_exists(path):
                return _result(False, None, "Directory does not exist.")
    else:
        # For files
        if valid_extensions:
            if not any(_has_extension(value, ext) for ext in valid_extensions):
                return _result(False, None, f"File must have one of these extensions: {', '.join(valid_extensions)}")

        if is_output:
            if path.parent != Path('.') and not _parent_exists(path):
                return _result(False, None, "Parent folder does not exist.")
        else:
            if not path.exists():
                return _result(False, None, "File does not exist")

    return _result(True, None, "")
