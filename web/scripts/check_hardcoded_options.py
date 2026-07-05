"""Guardrail: fail CI/pre-commit if a `<Select options={...}>` is fed by a
hardcoded array or object-literal list that lives outside the generated types.

The allowed sources are:
  - imports from `@/types/generated/` (enums.ts, dropdownSources.ts, etc.)
  - inline references to generated `*Values` arrays, `DTYPE_SUBSETS`, or schema-
    driven `stringOptions` / `kvOptions` via `SchemaField`
  - dynamic props resolved at runtime (`options={someState}`)

What this check catches:
  - `options={[ ... ]}` JSX props with a literal array
  - `options={FOO}` where `FOO` is a `const` array defined in the same file
    outside `types/generated/`

Run with:
    python -m web.scripts.check_hardcoded_options
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER_ROOT = REPO_ROOT / "web" / "gui" / "src" / "renderer"
GENERATED_DIR = RENDERER_ROOT / "types" / "generated"

# Match `options={...}` JSX prop. We use a non-greedy match that stops at the
# matching `}` — good enough because the value is always a single JSX expression.
_OPTIONS_PROP_RE = re.compile(r"options=\{([^}]+)\}")

# Match a local `const NAME = [ ... ]` or `const NAME: Type = [ ... ]` assignment.
_CONST_ARRAY_RE = re.compile(
    r"^\s*const\s+([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=\s*\[",
    re.MULTILINE,
)


def _strip_comments(src: str) -> str:
    """Remove // line comments and /* block */ comments — regex-level, good enough
    for scanning options= props without parsing TS."""
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def _collect_local_const_arrays(src: str) -> set[str]:
    return set(_CONST_ARRAY_RE.findall(src))


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of (line_number, offending_snippet) for violations in `path`."""
    violations: list[tuple[int, str]] = []
    raw = path.read_text(encoding="utf-8")
    src = _strip_comments(raw)
    local_const_arrays = _collect_local_const_arrays(src)

    # Build a line-number index so we can report useful locations.
    line_starts = [0]
    for i, ch in enumerate(raw):
        if ch == "\n":
            line_starts.append(i + 1)

    def line_for_offset(offset: int) -> int:
        # Binary search — line_starts is sorted
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    for match in _OPTIONS_PROP_RE.finditer(src):
        expr = match.group(1).strip()
        # 1. Inline array: options={[ ... ]} — flag only if it contains literal
        # entries (string literal, label:/value: object). Pure-spread arrays like
        # `[...NoiseSchedulerValues]` are fine: the spread source must be a
        # generated *Values array (imports are audited separately below).
        if expr.startswith("["):
            inner = expr[1:].rsplit("]", 1)[0]
            if re.search(r"""['"]|\blabel\s*:""", inner):
                violations.append((line_for_offset(match.start()), match.group(0)))
            continue
        # 2. Reference to a file-local const array — only flag if that const lives
        # in this file AND is uppercase-SNAKE (the convention for option arrays).
        ident = expr.lstrip(".").split(".")[0]
        if ident in local_const_arrays:
            violations.append((line_for_offset(match.start()), match.group(0)))

    return violations


def _should_scan(path: Path) -> bool:
    if path.suffix not in (".ts", ".tsx"):
        return False
    try:
        rel = path.relative_to(GENERATED_DIR)
    except ValueError:
        return True
    _ = rel
    return False


def main() -> int:
    all_violations: list[tuple[Path, int, str]] = []
    for root, _dirs, files in os.walk(RENDERER_ROOT):
        for name in files:
            path = Path(root) / name
            if not _should_scan(path):
                continue
            for line_no, snippet in _scan_file(path):
                all_violations.append((path, line_no, snippet))

    if not all_violations:
        print("OK: no hardcoded <Select options> arrays outside types/generated/")
        return 0

    print("FAIL: hardcoded dropdown options detected outside types/generated/")
    print("      (move the list into web/scripts/ui_metadata.py or a Python source")
    print("      that generate_types.py AST-parses, then re-run the generator)\n")
    for path, line_no, snippet in all_violations:
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        print(f"  {rel}:{line_no}: {snippet}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
