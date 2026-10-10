"""Collect translatable UI strings and merge them into resources/i18n/<lang>.json.

Usage: python scripts/i18n_extract.py [lang ...]   (default: zh_TW)

Existing translations are kept, new strings are added with an empty value, and
untranslated strings that no longer appear in the UI are dropped. Run it after merging
upstream changes to find strings that still need translating.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOBS = ["modules/ui/*.py"]
I18N_DIR = ROOT / "resources" / "i18n"

# component function -> positional indices holding display text
POSITIONAL_TEXT = {
    "label": (3,),
    "button": (3,),
    "icon_button": (3,),
    "preset_menu_button": (3,),
    "double_progress": (3, 4),
    "layer_filter_entry": (5, 6),
}
TEXT_KEYWORDS = {
    "text", "tooltip", "preset_label", "preset_tooltip", "entry_tooltip", "regex_tooltip", "label_1", "label_2",
    "add_button_text", "add_button_tooltip",
}
# dict literals like {'title': ..., 'tooltip': ...} used to build labels in a loop
TEXT_DICT_KEYS = {"title", "tooltip"}
COMPONENT_FUNCTIONS = set(POSITIONAL_TEXT) | {"switch", "entry", "path_entry", "options", "options_kv", "options_adv"}


def _str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
        return node.value
    if isinstance(node, ast.JoinedStr):
        # f'Text Encoder {i} Learning Rate' becomes the template key 'Text Encoder {i} Learning Rate'
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value.replace("{", "{{").replace("}", "}}"))
            elif (isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name)
                  and value.conversion == -1 and value.format_spec is None):
                parts.append("{" + value.value.id + "}")
            else:
                return None
        return "".join(parts)
    return None


def _func_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def extract(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in TEXT_DICT_KEYS:
                    found.append(_str(value))
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _func_name(node)
        if name == "t" and node.args:
            found.append(_str(node.args[0]))
        if name in COMPONENT_FUNCTIONS:
            found.extend(_str(node.args[i]) for i in POSITIONAL_TEXT.get(name, ()) if i < len(node.args))
            found.extend(_str(kw.value) for kw in node.keywords if kw.arg in TEXT_KEYWORDS)
    return [s for s in found if s]


def main():
    languages = sys.argv[1:] or ["zh_TW"]
    strings: dict[str, None] = {}
    for pattern in SOURCE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            for s in extract(path):
                strings.setdefault(s)

    I18N_DIR.mkdir(parents=True, exist_ok=True)
    for language in languages:
        out = I18N_DIR / f"{language}.json"
        existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        merged = {s: existing.get(s, "") for s in strings}
        # keep manually added keys for strings built at runtime
        for key, value in existing.items():
            if key not in merged and value:
                merged[key] = value
        out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        missing = sum(1 for v in merged.values() if not v)
        print(f"{out.relative_to(ROOT)}: {len(merged)} strings, {missing} untranslated")


if __name__ == "__main__":
    main()
