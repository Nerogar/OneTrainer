from __future__ import annotations

from collections.abc import Callable


def validate_resolution() -> Callable[[str], str | None]:
    """Return a resolution validator."""

    def _check(value: str) -> str | None:
        value = value.strip()
        if not value:
            return None

        dims = []

        if 'x' in value:
            parts = value.split('x')
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                dims = [int(parts[0].strip()), int(parts[1].strip())]
            else:
                return "Invalid format. Use <width>x<height> (e.g., 1024x768)"

        else:
            parts = value.split(',')
            if all(p.strip().isdigit() for p in parts):
                dims = [int(p.strip()) for p in parts]
            else:
                return "Must be a single integer, <width>x<height>, or comma-separated integers"

        for d in dims:
            if d <= 0:
                return f"Resolution cannot be less than or equal to 0 (found {d})."

        return None

    return _check


def check_range(
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
    message: str | None = None,
) -> Callable[[str], str | None]:
    """Validate that a numeric value falls within specified range, by default both bounds are inclusive."""

    def _check(value: str) -> str | None:
        try:
            v = float(value)
        except (ValueError, TypeError):
            return None  # type checking is handled by baseline validation

        if lower is not None:
            if lower_inclusive and v < lower:
                return message or f"Value must be at least {lower}"
            if not lower_inclusive and v <= lower:
                return message or f"Value must be greater than {lower}"

        if upper is not None:
            if upper_inclusive and v > upper:
                return message or f"Value must be at most {upper}"
            if not upper_inclusive and v >= upper:
                return message or f"Value must be less than {upper}"

        return None

    return _check


def compose(*checks: Callable[[str], str | None]) -> Callable[[str], str | None]:
    """Chain multiple ``extra_validate`` checks; return the first error.

    Usage::

        extra_validate=compose(
            check_range(lower=0, upper=1),
            <some other check>,
        )
    """

    def _check(value: str) -> str | None:
        for fn in checks:
            err = fn(value)
            if err is not None:
                return err
        return None

    return _check
