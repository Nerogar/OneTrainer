from __future__ import annotations

from collections.abc import Callable


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
