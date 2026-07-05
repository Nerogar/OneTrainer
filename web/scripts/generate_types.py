import ast
import importlib
import inspect
import json
import os
import re
import sys
from enum import Enum
from typing import Any, get_args, get_origin

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from modules.util.config.BaseConfig import BaseConfig
from modules.util.type_util import issubclass_safe

_incomplete_generation = False

# --- Prettier-compliant emit helpers -----------------------------------------
# These produce TypeScript that Prettier (printWidth 120, double quotes,
# trailingComma: all, quoteProps: as-needed) treats as a no-op, so the generated
# files stay consistent with the project's formatter without a post-format pass.

_PRETTIER_WIDTH = 120

_JS_RESERVED = frozenset({
    "break", "case", "catch", "class", "const", "continue", "debugger", "default",
    "delete", "do", "else", "enum", "export", "extends", "false", "finally", "for",
    "function", "if", "import", "in", "instanceof", "new", "null", "return", "super",
    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with",
    "yield", "let", "static", "implements", "interface", "package", "private",
    "protected", "public", "await", "async",
})

_JS_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def _js_key(key: str) -> str:
    """Quote object-literal keys only when required (matches Prettier quoteProps: as-needed)."""
    if _JS_IDENT_RE.match(key) and key not in _JS_RESERVED:
        return key
    return json.dumps(key)


def _ts_string(s: str) -> str:
    """Quote a string the way Prettier does: prefer double quotes, switch to single when the
    string contains more double quotes than single (to minimise backslash escapes)."""
    double = s.count('"')
    single = s.count("'")
    if double > single:
        # Emit single-quoted with escaping for backslashes and single quotes.
        escaped = s.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return json.dumps(s, ensure_ascii=False)


def _ts_number(value: float) -> str:
    """Format a number the way Prettier normalises it (strip leading zero in exponent, keep int-like
    floats with trailing .0 exactly as JSON emits them)."""
    rendered = json.dumps(value)
    # Prettier normalises `1e-08` -> `1e-8`, `3e+06` -> `3e6`, `2.5e+01` -> `2.5e1`.
    return re.sub(r"e([+-]?)0*(\d)", r"e\1\2", rendered).replace("e+", "e")


def _emit_entry(key: str, value: str, indent: str = "  ") -> list[str]:
    """Emit a single object-literal `key: value,` entry, wrapping if the line exceeds 120 chars."""
    single = f"{indent}{_js_key(key)}: {value},"
    if len(single) <= _PRETTIER_WIDTH:
        return [single]
    return [f"{indent}{_js_key(key)}:", f"{indent}  {value},"]


def _emit_array(items: list[str], prefix: str, indent: str = "") -> list[str]:
    """Emit an array literal — single-line if the full `{prefix}[{items}]` fits, else multi-line.
    `indent` is the whitespace prefix on the line where `prefix` starts; items are indented +2."""
    single = f"{prefix}[{', '.join(items)}]"
    if len(single) <= _PRETTIER_WIDTH:
        return [single]
    lines = [f"{prefix}["]
    item_indent = indent + "  "
    lines.extend(f"{item_indent}{item}," for item in items)
    lines.append(f"{indent}]")
    return lines


def _emit_union_type(name: str, values: list[str]) -> list[str]:
    """Emit `export type Name = "a" | "b";` on one line if it fits, else multi-line with leading pipes."""
    quoted = [_ts_string(v) for v in values]
    single = f"export type {name} = {' | '.join(quoted)};"
    if len(single) <= _PRETTIER_WIDTH:
        return [single]
    lines = [f"export type {name} ="]
    for i, q in enumerate(quoted):
        suffix = ";" if i == len(quoted) - 1 else ""
        lines.append(f"  | {q}{suffix}")
    return lines


def _emit_values_const(name: str, values: list[str]) -> list[str]:
    """Emit `export const NameValues: Name[] = [...];` single-line if it fits, else multi-line."""
    quoted = [_ts_string(v) for v in values]
    prefix = f"export const {name}Values: {name}[] = "
    block = _emit_array(quoted, prefix)
    if len(block) == 1:
        return [f"{block[0]};"]
    block[-1] = f"{block[-1]};"
    return block

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "web", "gui", "src", "renderer", "types", "generated"
)

ENUM_DIR = os.path.join(PROJECT_ROOT, "modules", "util", "enum")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "modules", "util", "config")

_KNOWN_ENUM_MODULES = {
    "modules.util.enum.AudioFormat",
    "modules.util.enum.DataType",
    "modules.util.enum.ModelType",
    "modules.util.enum.Optimizer",
    "modules.util.enum.TrainingMethod",
}

_KNOWN_CONFIG_CLASSES = {
    "TrainConfig", "TrainOptimizerConfig", "TrainModelPartConfig",
    "TrainEmbeddingConfig", "QuantizationConfig", "ConceptConfig",
    "ConceptImageConfig", "ConceptTextConfig", "SampleConfig",
    "CloudConfig", "CloudSecretsConfig", "SecretsConfig",
}


def discover_enum_modules() -> list[str]:
    modules = []
    for filename in sorted(os.listdir(ENUM_DIR)):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        module_path = f"modules.util.enum.{module_name}"
        modules.append(module_path)
    return modules


def discover_config_classes() -> list[tuple[str, type]]:
    configs = []
    seen = set()
    for filename in sorted(os.listdir(CONFIG_DIR)):
        if filename.startswith("_") or not filename.endswith(".py"):
            continue
        module_name = filename[:-3]
        module_path = f"modules.util.config.{module_name}"
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass_safe(obj, BaseConfig)
                and obj is not BaseConfig
                and hasattr(obj, "default_values")
                and attr_name not in seen
            ):
                seen.add(attr_name)
                configs.append((attr_name, obj))
    return configs


ENUM_MODULES = discover_enum_modules()

_missing_enums = _KNOWN_ENUM_MODULES - set(ENUM_MODULES)
if _missing_enums:
    raise RuntimeError(f"Dynamic enum scan missed known modules: {_missing_enums}")


def collect_enums() -> list[tuple[str, type]]:
    enums = []
    for module_path in ENUM_MODULES:
        mod = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Enum) and obj is not Enum:
                enums.append((name, obj))
    return enums


def collect_configs() -> list[tuple[str, type]]:
    configs = discover_config_classes()
    found_names = {name for name, _ in configs}
    missing = _KNOWN_CONFIG_CLASSES - found_names
    if missing:
        raise RuntimeError(f"Dynamic config scan missed known classes: {missing}")
    return configs


def python_type_to_ts(py_type: type, nullable: bool, enum_names: set[str]) -> str:
    if py_type is str:
        ts = "string"
    elif py_type is bool:
        ts = "boolean"
    elif py_type is int or py_type is float:
        ts = "number"
    elif py_type is dict:
        ts = "Record<string, unknown>"
    elif issubclass_safe(py_type, Enum) or issubclass_safe(py_type, BaseConfig):
        ts = py_type.__name__
    elif py_type is list or get_origin(py_type) is list:
        args = get_args(py_type)
        if args:
            inner = args[0]
            if issubclass_safe(inner, BaseConfig) or issubclass_safe(inner, Enum):
                ts = f"{inner.__name__}[]"
            elif inner is str:
                ts = "string[]"
            elif inner is int or inner is float:
                ts = "number[]"
            elif inner is bool:
                ts = "boolean[]"
            elif get_origin(inner) is dict:
                dict_args = get_args(inner)
                if dict_args and len(dict_args) == 2:
                    key_ts = python_type_to_ts(dict_args[0], False, enum_names)
                    val_ts = python_type_to_ts(dict_args[1], False, enum_names)
                    # array-type: array-simple requires Array<T> when T is a complex type
                    ts = f"Array<Record<{key_ts}, {val_ts}>>"
                else:
                    ts = "Array<Record<string, unknown>>"
            else:
                ts = "unknown[]"
        else:
            ts = "unknown[]"
    elif get_origin(py_type) is dict:
        args = get_args(py_type)
        if args and len(args) == 2:
            key_ts = python_type_to_ts(args[0], False, enum_names)
            val_ts = python_type_to_ts(args[1], False, enum_names)
            ts = f"Record<{key_ts}, {val_ts}>"
        else:
            ts = "Record<string, unknown>"
    else:
        ts = "unknown"

    if nullable:
        ts = f"{ts} | null"

    return ts


def generate_enums_ts(enums: list[tuple[str, type]]) -> str:
    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend enums change.",
        "",
    ]

    for name, enum_cls in sorted(enums, key=lambda x: x[0]):
        values = [member.value for member in enum_cls]
        lines.extend(_emit_union_type(name, values))
        lines.append("")
        lines.extend(_emit_values_const(name, values))
        lines.append("")

    return "\n".join(lines)


def generate_config_ts(
    configs: list[tuple[str, type]], enum_names: set[str]
) -> str:
    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend config classes change.",
        "",
        "import type {",
    ]

    used_enums = set()
    for _class_name, cls in configs:
        instance = cls.default_values()
        for field_type in instance.types.values():
            if issubclass_safe(field_type, Enum):
                used_enums.add(field_type.__name__)
            elif get_origin(field_type) is list:
                args = get_args(field_type)
                if args and issubclass_safe(args[0], Enum):
                    used_enums.add(args[0].__name__)
            elif get_origin(field_type) is dict:
                args = get_args(field_type)
                if args:
                    for arg in args:
                        if issubclass_safe(arg, Enum):
                            used_enums.add(arg.__name__)

    # simple-import-sort/imports uses case-insensitive ordering within a named-import block.
    lines.extend(f"  {enum_name}," for enum_name in sorted(used_enums, key=str.lower))
    lines.append('} from "./enums";')
    lines.append("")

    generated = set()

    def generate_interface(class_name: str, cls: type):
        if class_name in generated:
            return ""
        generated.add(class_name)

        result_lines = []

        instance = cls.default_values()
        for field_type in instance.types.values():
            if issubclass_safe(field_type, BaseConfig) and field_type.__name__ not in generated:
                for cn, cc in configs:
                    if cn == field_type.__name__:
                        dep = generate_interface(cn, cc)
                        if dep:
                            result_lines.append(dep)
                        break
            elif get_origin(field_type) is list:
                args = get_args(field_type)
                if args and issubclass_safe(args[0], BaseConfig) and args[0].__name__ not in generated:
                    for cn, cc in configs:
                        if cn == args[0].__name__:
                            dep = generate_interface(cn, cc)
                            if dep:
                                result_lines.append(dep)
                            break
            elif get_origin(field_type) is dict:
                args = get_args(field_type)
                if args and len(args) > 1 and issubclass_safe(args[1], BaseConfig) and args[1].__name__ not in generated:
                    for cn, cc in configs:
                        if cn == args[1].__name__:
                            dep = generate_interface(cn, cc)
                            if dep:
                                result_lines.append(dep)
                            break

        iface_lines = [f"export interface {class_name} {{"]

        for field_name in instance.types:
            field_type = instance.types[field_name]
            nullable = instance.nullables.get(field_name, False)
            ts_type = python_type_to_ts(field_type, nullable, enum_names)
            iface_lines.append(f"  {field_name}: {ts_type};")

        iface_lines.append("}")
        iface_lines.append("")

        result_lines.append("\n".join(iface_lines))
        return "\n".join(result_lines)

    for class_name, cls in configs:
        result = generate_interface(class_name, cls)
        if result:
            lines.append(result)

    return "\n".join(lines)


def generate_metadata_ts(
    configs: list[tuple[str, type]], enum_names: set[str]
) -> str:
    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend config classes change.",
        "",
        "export interface FieldMetadata {",
        '  type: "string" | "number" | "boolean" | "enum" | "config" | "list" | "dict";',
        "  nullable: boolean;",
        "  enumType?: string;",
        "  configType?: string;",
        "  defaultValue: unknown;",
        "}",
        "",
        "export type ConfigMetadata = Record<string, FieldMetadata>;",
        "",
    ]

    for class_name, cls in configs:
        instance = cls.default_values()
        lines.append(f"export const {class_name}Metadata: ConfigMetadata = {{")

        for field_name in instance.types:
            field_type = instance.types[field_name]
            nullable = instance.nullables.get(field_name, False)
            default_value = instance.default_values.get(field_name)

            if field_type is str:
                meta_type = "string"
                enum_type = None
                config_type = None
            elif field_type is bool:
                meta_type = "boolean"
                enum_type = None
                config_type = None
            elif field_type is int or field_type is float:
                meta_type = "number"
                enum_type = None
                config_type = None
            elif issubclass_safe(field_type, Enum):
                meta_type = "enum"
                enum_type = field_type.__name__
                config_type = None
            elif issubclass_safe(field_type, BaseConfig):
                meta_type = "config"
                enum_type = None
                config_type = field_type.__name__
            elif field_type is list or get_origin(field_type) is list:
                meta_type = "list"
                enum_type = None
                config_type = None
                args = get_args(field_type)
                if args and issubclass_safe(args[0], BaseConfig):
                    config_type = args[0].__name__
            elif field_type is dict or get_origin(field_type) is dict:
                meta_type = "dict"
                enum_type = None
                config_type = None
                args = get_args(field_type)
                if args and len(args) > 1 and issubclass_safe(args[1], BaseConfig):
                    config_type = args[1].__name__
            else:
                meta_type = "string"
                enum_type = None
                config_type = None

            if default_value is None:
                default_json = "null"
            elif isinstance(default_value, bool):
                default_json = "true" if default_value else "false"
            elif isinstance(default_value, (int, float)):
                if default_value == float("inf"):
                    default_json = '"Infinity"'
                elif default_value == float("-inf"):
                    default_json = '"-Infinity"'
                else:
                    default_json = _ts_number(default_value)
            elif isinstance(default_value, str):
                default_json = _ts_string(default_value)
            elif isinstance(default_value, Enum):
                default_json = json.dumps(str(default_value))
            elif isinstance(default_value, BaseConfig):
                default_json = "null"
            elif isinstance(default_value, list):
                default_json = "[]"
            elif isinstance(default_value, dict):
                default_json = "{}"
            else:
                default_json = "null"

            parts = [f'type: "{meta_type}"', f"nullable: {str(nullable).lower()}"]
            if enum_type:
                parts.append(f'enumType: "{enum_type}"')
            if config_type:
                parts.append(f'configType: "{config_type}"')
            parts.append(f"defaultValue: {default_json}")

            single = f"  {_js_key(field_name)}: {{ {', '.join(parts)} }},"
            if len(single) <= _PRETTIER_WIDTH:
                lines.append(single)
            else:
                lines.append(f"  {_js_key(field_name)}: {{")
                lines.extend(f"    {part}," for part in parts)
                lines.append("  },")

        lines.append("};")
        lines.append("")

    return "\n".join(lines)


def _safe_call_method(obj: object, method_name: str) -> bool:
    try:
        return bool(getattr(obj, method_name)())
    except TypeError:
        return False


def generate_model_type_info_ts(enums: list[tuple[str, type]]) -> str:
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend ModelType or TopBar logic changes.",
        "",
        'import type { ModelType, TrainingMethod } from "./enums";',
        "",
    ]

    # inspect.getmembers doesn't find these due to Enum metaclass; use __dict__
    group_methods = []
    for name, obj in ModelType.__dict__.items():
        if name.startswith(("is_", "has_")) and callable(obj):
            group_methods.append(name)
    group_methods.sort()

    lines.append("/** Model type groupings derived from ModelType.is_*() / has_*() methods. */")
    lines.append("export const MODEL_TYPE_GROUPS: Record<string, ModelType[]> = {")
    for method_name in group_methods:
        members_in_group = [
            mt.value for mt in ModelType if _safe_call_method(mt, method_name)
        ]
        if members_in_group:
            quoted = [json.dumps(m) for m in members_in_group]
            prefix = f"  {_js_key(method_name)}: "
            block = _emit_array(quoted, prefix, indent="  ")
            block[-1] = f"{block[-1]},"
            lines.extend(block)
    lines.append("};")
    lines.append("")

    lines.append("/** Reverse lookup: for any ModelType, which groups it belongs to. */")
    lines.append("export const MODEL_TYPE_FLAGS: Record<ModelType, string[]> = {")
    for mt in ModelType:
        flags = [
            method_name for method_name in group_methods
            if _safe_call_method(mt, method_name)
        ]
        quoted = [json.dumps(f) for f in flags]
        prefix = f"  {_js_key(mt.value)}: "
        block = _emit_array(quoted, prefix, indent="  ")
        block[-1] = f"{block[-1]},"
        lines.extend(block)
    lines.append("};")
    lines.append("")

    lines.append("/** Allowed training methods per model type (from TopBar.py). */")
    lines.append("export const TRAINING_METHODS_BY_MODEL: Record<ModelType, TrainingMethod[]> = {")
    for mt in ModelType:
        if mt.is_stable_diffusion():
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING, TrainingMethod.FINE_TUNE_VAE]
        elif (mt.is_stable_diffusion_3() or mt.is_stable_diffusion_xl() or mt.is_wuerstchen()
              or mt.is_pixart() or mt.is_flux_1() or mt.is_sana()
              or mt.is_hunyuan_video() or mt.is_hi_dream() or mt.is_chroma()):
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING]
        elif mt.is_qwen() or mt.is_z_image() or mt.is_flux_2() or mt.is_ernie():
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA]
        else:
            methods = [TrainingMethod.FINE_TUNE, TrainingMethod.LORA, TrainingMethod.EMBEDDING]
        quoted = [json.dumps(m.value) for m in methods]
        prefix = f"  {_js_key(mt.value)}: "
        block = _emit_array(quoted, prefix, indent="  ")
        block[-1] = f"{block[-1]},"
        lines.extend(block)
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def _find_layer_presets_attr(cls: type) -> dict | None:
    for ancestor in cls.__mro__:
        if "LAYER_PRESETS" in ancestor.__dict__:
            value = ancestor.__dict__["LAYER_PRESETS"]
            if isinstance(value, dict):
                return value
    return None


def _normalize_layer_preset_value(value: Any) -> dict[str, Any]:
    # BaseZImageSetup uses {'patterns': [...], 'regex': True}; others use list[str].
    if isinstance(value, dict):
        patterns = value.get("patterns", []) or []
        return {"patterns": [str(p) for p in patterns], "regex": bool(value.get("regex", False))}
    if isinstance(value, (list, tuple)):
        return {"patterns": [str(p) for p in value], "regex": False}
    return {"patterns": [str(value)], "regex": False}


def generate_layer_presets_ts() -> str:
    """Emit LAYER_PRESETS_BY_MODEL derived from Base*Setup.LAYER_PRESETS.

    Keyed by ModelType — presets live on the Base*Setup family class and are
    inherited by every LoRA/FineTune/Embedding subclass, so TrainingMethod is
    irrelevant here. Mirrors CTk behavior in modules/ui/TrainingTab.py and
    modules/ui/ModelTab.py which both read `cls.LAYER_PRESETS` from
    `create.get_model_setup_class(model_type, training_method)`.
    """
    global _incomplete_generation

    from modules.util.enum.ModelType import ModelType

    presets_by_model: dict[str, dict[str, dict[str, Any]]] = {}

    try:
        from modules.modelSetup.BaseModelSetup import BaseModelSetup
        from modules.util import create as _create_side_effects  # noqa: F401
        from modules.util import factory

        registry = factory.__dict__.get("__registry", {}).get(BaseModelSetup, [])
        seen: set[ModelType] = set()
        for args, _kwargs, cls in registry:
            if not args:
                continue
            mt = args[0]
            if not isinstance(mt, ModelType) or mt in seen:
                continue
            raw = _find_layer_presets_attr(cls)
            if raw is None:
                continue
            presets_by_model[mt.value] = {
                str(k): _normalize_layer_preset_value(v) for k, v in raw.items()
            }
            seen.add(mt)
    except Exception as e:
        print(f"  WARNING: Could not enumerate LAYER_PRESETS from factory registry ({e})")
        print("  Layer preset map will be empty.")
        print("  Run from the OneTrainer venv for complete output: python -m web.scripts.generate_types")
        _incomplete_generation = True

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend Base*Setup.LAYER_PRESETS change.",
        "",
        'import type { ModelType } from "./enums";',
        "",
        "export interface LayerPresetDef {",
        "  patterns: string[];",
        "  regex: boolean;",
        "}",
        "",
        "export type LayerPresetMap = Record<string, LayerPresetDef>;",
        "",
        "/** Fallback preset map used when a ModelType has no registered BaseSetup. */",
        "export const EMPTY_PRESETS: LayerPresetMap = {",
        "  full: { patterns: [], regex: false },",
        "};",
        "",
        "/**",
        " * Per-ModelType layer-filter presets sourced from Base*Setup.LAYER_PRESETS.",
        " * Selecting a preset should write both the joined patterns to layer_filter",
        " * and the regex flag to layer_filter_regex, matching the CTk UI behavior.",
        " */",
        "export const LAYER_PRESETS_BY_MODEL: Partial<Record<ModelType, LayerPresetMap>> = {",
    ]

    for mt_value in sorted(presets_by_model.keys()):
        preset_map = presets_by_model[mt_value]
        lines.append(f"  {_js_key(mt_value)}: {{")
        for preset_name in preset_map:
            defn = preset_map[preset_name]
            quoted_patterns = [json.dumps(p) for p in defn["patterns"]]
            regex_str = "true" if defn["regex"] else "false"
            patterns_inline = f"[{', '.join(quoted_patterns)}]"
            value = f"{{ patterns: {patterns_inline}, regex: {regex_str} }}"
            single = f"    {_js_key(preset_name)}: {value},"
            if len(single) <= _PRETTIER_WIDTH:
                lines.append(single)
            else:
                lines.append(f"    {_js_key(preset_name)}: {{")
                patterns_prefix = "      patterns: "
                patterns_block = _emit_array(quoted_patterns, patterns_prefix, indent="      ")
                patterns_block[-1] = f"{patterns_block[-1]},"
                lines.extend(patterns_block)
                lines.append(f"      regex: {regex_str},")
                lines.append("    },")
        lines.append("  },")

    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_loss_weight_info_ts() -> str:
    """Emit a map of LossWeight -> supports_flow_matching boolean.

    Mirrors CTk behavior in modules/ui/TrainingTab.py, which filters the
    loss_weight_fn dropdown so only entries whose supports_flow_matching()
    matches the current model's is_flow_matching() (plus CONSTANT) are shown.
    """
    from modules.util.enum.LossWeight import LossWeight

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when LossWeight.supports_flow_matching changes.",
        "",
        'import type { LossWeight } from "./enums";',
        "",
        "/**",
        " * Maps each LossWeight value to whether it supports flow-matching models.",
        " * Sourced from LossWeight.supports_flow_matching() in",
        " * modules/util/enum/LossWeight.py. Used by the schema renderer to filter",
        " * the loss_weight_fn dropdown per the current model_type.",
        " */",
        "export const LOSS_WEIGHT_SUPPORTS_FLOW_MATCHING: Record<LossWeight, boolean> = {",
    ]

    for member in LossWeight:
        flag = "true" if member.supports_flow_matching() else "false"
        lines.append(f"  {_js_key(member.value)}: {flag},")

    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_optimizer_info_ts(enums: list[tuple[str, type]]) -> str:
    from modules.util.enum.Optimizer import Optimizer

    # optimizer_util has a circular import; extract defaults from source via AST
    global _incomplete_generation
    try:
        OPTIMIZER_DEFAULT_PARAMETERS = _extract_optimizer_defaults()
    except Exception as e:
        print(f"  WARNING: Could not extract OPTIMIZER_DEFAULT_PARAMETERS ({e})")
        print("  Optimizer defaults will be empty.")
        print("  Run from the OneTrainer venv for complete output: python -m web.scripts.generate_types")
        OPTIMIZER_DEFAULT_PARAMETERS = {}
        _incomplete_generation = True

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when backend optimizer definitions change.",
        "",
        'import type { Optimizer } from "./enums";',
        "",
    ]

    adaptive = [opt for opt in Optimizer if opt.is_adaptive]
    schedule_free = [opt for opt in Optimizer if opt.is_schedule_free]
    fused_back_pass = [opt for opt in Optimizer if opt.supports_fused_back_pass()]

    def opt_array(name: str, description: str, opts: list) -> None:
        lines.append(f"/** {description} */")
        quoted = [_ts_string(opt.value) for opt in opts]
        prefix = f"export const {name}: Optimizer[] = "
        block = _emit_array(quoted, prefix)
        if len(block) == 1:
            lines.append(f"{block[0]};")
        else:
            block[-1] = f"{block[-1]};"
            lines.extend(block)
        lines.append("")

    opt_array("ADAPTIVE_OPTIMIZERS", "Optimizers with adaptive learning rates.", adaptive)
    opt_array("SCHEDULE_FREE_OPTIMIZERS", "Schedule-free optimizers.", schedule_free)
    opt_array("FUSED_BACK_PASS_OPTIMIZERS", "Optimizers that support fused backward pass.", fused_back_pass)

    # Prettier reformats a Record<K, { ... multi-line inline object ... }> into this 5-line
    # wrapper form, so emit that shape directly.
    lines.append("/** Per-optimizer boolean property flags. */")
    lines.append("export const OPTIMIZER_FLAGS: Record<")
    lines.append("  Optimizer,")
    lines.append("  {")
    lines.append("    isAdaptive: boolean;")
    lines.append("    isScheduleFree: boolean;")
    lines.append("    supportsFusedBackPass: boolean;")
    lines.append("  }")
    lines.append("> = {")
    for opt in Optimizer:
        a = "true" if opt.is_adaptive else "false"
        sf = "true" if opt.is_schedule_free else "false"
        fb = "true" if opt.supports_fused_back_pass() else "false"
        lines.append(f"  {_js_key(opt.value)}: {{ isAdaptive: {a}, isScheduleFree: {sf}, supportsFusedBackPass: {fb} }},")
    lines.append("};")
    lines.append("")

    def serialize_value(v: Any) -> str:
        if v is None:
            return "null"
        elif isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, (int, float)):
            if v == float("inf"):
                return "Infinity"
            elif v == float("-inf"):
                return "-Infinity"
            return _ts_number(v)
        elif isinstance(v, str):
            return _ts_string(v)
        elif isinstance(v, dict):
            if not v:
                return "{}"
            pairs = ", ".join(f"{_js_key(str(k))}: {serialize_value(vv)}" for k, vv in v.items())
            return f"{{ {pairs} }}"
        elif isinstance(v, list):
            if not v:
                return "[]"
            items = ", ".join(serialize_value(item) for item in v)
            return f"[{items}]"
        elif isinstance(v, Enum):
            return _ts_string(str(v))
        else:
            return _ts_string(str(v))

    lines.append("/** Default parameter values per optimizer (from optimizer_util.py). */")
    lines.append("export const OPTIMIZER_DEFAULTS: Record<Optimizer, Record<string, unknown>> = {")
    for opt in Optimizer:
        defaults = OPTIMIZER_DEFAULT_PARAMETERS.get(opt, {})
        if not defaults:
            lines.append(f"  {_js_key(opt.value)}: {{}},")
            continue
        lines.append(f"  {_js_key(opt.value)}: {{")
        for key, val in defaults.items():
            lines.extend(_emit_entry(key, serialize_value(val), indent="    "))
        lines.append("  },")
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def _safe_literal_eval(node: ast.AST) -> Any:
    """Like ast.literal_eval but additionally supports float('inf'), float('-inf'), and float('nan').

    Plain ast.literal_eval rejects Call nodes, so any default like
    `"growth_rate": float('inf')` causes silent fallback to {} during extraction.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal_eval(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_safe_literal_eval(e) for e in node.elts]
    if isinstance(node, ast.Set):
        return {_safe_literal_eval(e) for e in node.elts}
    if isinstance(node, ast.Dict):
        if len(node.keys) != len(node.values):
            raise ValueError("malformed dict literal")
        return {_safe_literal_eval(k): _safe_literal_eval(v) for k, v in zip(node.keys, node.values, strict=True)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _safe_literal_eval(node.operand)
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and not node.keywords
    ):
        arg = _safe_literal_eval(node.args[0])
        if isinstance(arg, str):
            return float(arg)
        raise ValueError(f"unsupported float() argument: {arg!r}")
    raise ValueError(f"unsupported AST node: {ast.dump(node)}")


def _extract_dict_from_source(source: str, variable_name: str) -> dict:
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        return _safe_literal_eval(node.value)
        return {}
    except (SyntaxError, ValueError) as e:
        print(f"  WARNING: Could not parse {variable_name}: {e}")
        return {}


def _extract_optimizer_defaults() -> dict:
    """AST-based extraction to avoid circular import in optimizer_util."""
    from modules.util.enum.Optimizer import Optimizer

    source_path = os.path.join(PROJECT_ROOT, "modules", "util", "optimizer_util.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  WARNING: Could not parse optimizer_util.py: {e}")
        return {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OPTIMIZER_DEFAULT_PARAMETERS":
                    if not isinstance(node.value, ast.Dict):
                        continue
                    result = {}
                    for key_node, val_node in zip(node.value.keys, node.value.values, strict=False):
                        if isinstance(key_node, ast.Attribute) and isinstance(key_node.value, ast.Name):
                            if key_node.value.id == "Optimizer":
                                try:
                                    opt_member = Optimizer[key_node.attr]
                                except KeyError:
                                    continue
                                try:
                                    val = _safe_literal_eval(val_node)
                                except (ValueError, SyntaxError) as e:
                                    print(f"  WARNING: Could not extract defaults for Optimizer.{key_node.attr}: {e}")
                                    val = {}
                                result[opt_member] = val
                    return result

    print("  WARNING: Could not find OPTIMIZER_DEFAULT_PARAMETERS in optimizer_util.py")
    return {}


def generate_optimizer_defaults_json(enums: list[tuple[str, type]]) -> str:
    from modules.util.enum.Optimizer import Optimizer

    global _incomplete_generation
    try:
        OPTIMIZER_DEFAULT_PARAMETERS = _extract_optimizer_defaults()
    except Exception as e:
        print(f"  WARNING: Could not extract OPTIMIZER_DEFAULT_PARAMETERS ({e})")
        OPTIMIZER_DEFAULT_PARAMETERS = {}
        _incomplete_generation = True

    def clean_value(v: Any) -> Any:
        if v is None:
            return None
        elif isinstance(v, bool):
            return v
        elif isinstance(v, (int, float)):
            if v == float("inf"):
                return "Infinity"
            elif v == float("-inf"):
                return "-Infinity"
            return v
        elif isinstance(v, str):
            return v
        elif isinstance(v, dict):
            return {str(k): clean_value(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [clean_value(item) for item in v]
        elif isinstance(v, Enum):
            return str(v)
        else:
            return str(v)

    result: dict[str, dict] = {}
    for opt in Optimizer:
        defaults = OPTIMIZER_DEFAULT_PARAMETERS.get(opt, {})
        result[str(opt)] = {str(k): clean_value(v) for k, v in defaults.items()}

    return json.dumps(result, indent=2)


def _extract_key_detail_map() -> dict:
    """AST-based extraction to avoid importing tkinter-dependent UI module."""
    source_path = os.path.join(PROJECT_ROOT, "modules", "ui", "OptimizerParamsWindow.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    return _extract_dict_from_source(source, "KEY_DETAIL_MAP")


def generate_optimizer_key_details_json() -> str:
    key_detail_map = _extract_key_detail_map()
    return json.dumps(key_detail_map, indent=2)


def generate_optimizer_key_details_ts() -> str:
    key_detail_map = _extract_key_detail_map()

    # Derive the type union from actual values so new types (e.g. enum-like
    # 'CenteredWDMode') don't silently violate the type contract.
    type_values = sorted({str(detail["type"]) for detail in key_detail_map.values()})
    type_union = " | ".join(f'"{t}"' for t in type_values)

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Source: modules/ui/OptimizerParamsWindow.py KEY_DETAIL_MAP",
        "// Do not edit manually.",
        "",
        "export interface OptimizerKeyDetail {",
        "  title: string;",
        "  tooltip: string;",
        f"  type: {type_union};",
        "}",
        "",
        "export const OPTIMIZER_KEY_DETAILS: Record<string, OptimizerKeyDetail> = {",
    ]

    for key in sorted(key_detail_map.keys()):
        detail = key_detail_map[key]
        title = _ts_string(detail["title"])
        tooltip = _ts_string(detail["tooltip"])
        detail_type = _ts_string(detail["type"])
        inline = f"{{ title: {title}, tooltip: {tooltip}, type: {detail_type} }}"
        single = f"  {_js_key(key)}: {inline},"
        if len(single) <= _PRETTIER_WIDTH:
            lines.append(single)
        else:
            lines.append(f"  {_js_key(key)}: {{")
            lines.append(f"    title: {title},")
            lines.extend(_emit_entry("tooltip", tooltip, indent="    "))
            lines.append(f"    type: {detail_type},")
            lines.append("  },")

    lines.append("};")
    lines.append("")
    return "\n".join(lines)


_ACRONYMS = {
    "SDXL", "VAE", "LORA", "GAN", "FP16", "FP32", "BF16", "NF4",
    "CPU", "GPU", "TPU", "EMA", "LR", "GGUF", "BNBFP4", "BNBNF4",
    "RGB", "RGBA", "HDR", "SRT", "JSON", "CSV", "MP3", "MP4", "FLAC",
    "WAV", "OGG", "AVI", "MKV", "WEBM", "GIF", "PNG", "JPG", "JPEG",
    "WEBP", "BMP", "TIFF",
}

_SPECIAL_LABELS = {
    "ADAMW": "AdamW",
    "LORA": "LoRA",
    "ADAFACTOR": "Adafactor",
}

_VERSION_SUFFIXES = {"15": "1.5", "20": "2.0", "21": "2.1", "30": "3.0", "35": "3.5"}


def _auto_label(value: str) -> str:
    if value in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[value]

    parts = value.split("_")
    result = []
    for part in parts:
        if part in _ACRONYMS:
            result.append(part)
        elif part in _VERSION_SUFFIXES:
            result.append(_VERSION_SUFFIXES[part])
        elif part.isdigit():
            result.append(part)
        else:
            result.append(part.capitalize())
    return " ".join(result)


def _auto_tooltip(field_path: str) -> str:
    readable = field_path.replace(".", " ").replace("_", " ")
    return readable[:1].upper() + readable[1:] if readable else ""


def generate_enum_labels_ts(enums: list[tuple[str, type]]) -> str:
    from web.scripts.ui_metadata import ENUM_DISPLAY_LABELS

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Update labels in web/scripts/ui_metadata.py.",
        "",
    ]

    flat_labels: dict[str, str] = {}
    for enum_name, enum_cls in enums:
        overrides = ENUM_DISPLAY_LABELS.get(enum_name, {})
        for member in enum_cls:
            flat_labels[member.value] = overrides.get(member.value, _auto_label(member.value))

    lines.append("const labels: Record<string, string> = {")
    for value in sorted(flat_labels.keys()):
        lines.extend(_emit_entry(value, _ts_string(flat_labels[value])))
    lines.append("};")
    lines.append("")

    lines.append("function formatFallback(value: string): string {")
    lines.append("  return value")
    lines.append('    .replace(/_/g, " ")')
    lines.append("    .replace(")
    lines.append("      /\\b([A-Za-z])([A-Za-z]*)\\b/g,")
    lines.append(
        "      (_match, first: string, rest: string) => first.toUpperCase() + rest.toLowerCase(),"
    )
    lines.append("    );")
    lines.append("}")
    lines.append("")

    lines.append("/**")
    lines.append(" * Returns a human-friendly display label for any enum value string.")
    lines.append(" * Looks up the value in a curated map. Falls back to title-casing.")
    lines.append(" */")
    lines.append("export function enumLabel(value: string): string {")
    lines.append("  return labels[value] ?? formatFallback(value);")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_data_type_subsets_ts() -> str:
    from web.scripts.ui_metadata import DTYPE_SUBSETS

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Update subsets in web/scripts/ui_metadata.py.",
        "",
        'import type { DataType } from "./enums";',
        "",
        "export interface DTypeOption {",
        "  label: string;",
        "  value: DataType;",
        "}",
        "",
    ]

    lines.append("export const DTYPE_SUBSETS: Record<string, DTypeOption[]> = {")
    for subset_name, options in DTYPE_SUBSETS.items():
        lines.append(f"  {_js_key(subset_name)}: [")
        for label, value in options:
            lines.append(f"    {{ label: {json.dumps(label)}, value: {json.dumps(value)} }},")
        lines.append("  ],")
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_tooltips_ts(configs: list[tuple[str, type]]) -> str:
    from web.scripts.ui_metadata import FIELD_TOOLTIPS, WIDE_TOOLTIPS

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Update tooltips in web/scripts/ui_metadata.py.",
        "",
    ]

    all_tooltips: dict[str, str] = {}
    all_tooltips.update(FIELD_TOOLTIPS)

    for _class_name, cls in configs:
        try:
            inst = cls.default_values()
            for field_name in inst.to_dict():
                if field_name.startswith("__"):
                    continue
                if field_name not in all_tooltips:
                    auto = _auto_tooltip(field_name)
                    if auto:
                        all_tooltips[field_name] = auto
        except Exception:
            continue

    lines.append("/** Tooltip text for config fields, keyed by dot-notation field path. */")
    lines.append("export const FIELD_TOOLTIPS: Record<string, string> = {")
    for key in sorted(all_tooltips.keys()):
        tooltip = all_tooltips[key]
        lines.extend(_emit_entry(key, _ts_string(tooltip)))
    lines.append("};")
    lines.append("")

    lines.append("/** Field keys that require wide tooltip display. */")
    lines.append("export const WIDE_TOOLTIP_KEYS = new Set<string>([")
    lines.extend(f"  {json.dumps(key)}," for key in sorted(WIDE_TOOLTIPS))
    lines.append("]);")
    lines.append("")

    lines.append("/** Get tooltip text for a config field key. Returns undefined if not found. */")
    lines.append("export function getTooltip(fieldKey: string): string | undefined {")
    lines.append("  return FIELD_TOOLTIPS[fieldKey];")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# --- Dropdown sources ---------------------------------------------------------
# Any `<Select>` whose options are not covered by the enum / dtypeSubset / layer-
# preset pipelines is sourced here: CTk `components.options_kv(...)` calls for
# Convert & Concept modals (authoritative Python source), plus curated tool-modal
# lists from `ui_metadata.TOOL_DROPDOWN_OPTIONS`. The generator AST-parses the CTk
# calls, resolves `Enum.MEMBER` references via the runtime enum registry, and
# emits plain `{ label, value }[]` arrays.


def _resolve_ast_value_as_str(node: ast.AST, enum_lookup: dict[str, type]) -> str | None:
    """Resolve an AST expression to its string value.

    Handles two shapes used by CTk `options_kv` calls:
    - `ast.Constant` string literals → the literal value.
    - `ast.Attribute` of the form `EnumClass.MEMBER` → the member's `.value`.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        cls = enum_lookup.get(node.value.id)
        if cls is None:
            return None
        member = getattr(cls, node.attr, None)
        if member is None:
            return None
        val = getattr(member, "value", None)
        return val if isinstance(val, str) else None
    return None


def _extract_options_kv_calls(
    filepath: str, enum_lookup: dict[str, type]
) -> dict[str, list[tuple[str, str]]]:
    """AST-parse a CTk UI file for `options_kv(..., [(label, value), ...], state, "field")` calls.

    Returns `{field_name: [(label, value), ...]}`. Tuples whose label isn't a string
    literal or whose value can't be resolved are skipped.
    """
    with open(filepath, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    results: dict[str, list[tuple[str, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = (
            func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name)
            else None
        )
        if func_name != "options_kv":
            continue
        # components.options_kv(master, row, col, list, state, field_name, ...)
        if len(node.args) < 6:
            continue
        options_node, field_node = node.args[3], node.args[5]
        if not isinstance(options_node, ast.List):
            continue
        if not (isinstance(field_node, ast.Constant) and isinstance(field_node.value, str)):
            continue

        pairs: list[tuple[str, str]] = []
        for item in options_node.elts:
            if not (isinstance(item, ast.Tuple) and len(item.elts) == 2):
                continue
            label_node, value_node = item.elts
            if not (isinstance(label_node, ast.Constant) and isinstance(label_node.value, str)):
                continue
            value = _resolve_ast_value_as_str(value_node, enum_lookup)
            if value is None:
                continue
            pairs.append((label_node.value, value))

        if pairs:
            results[field_node.value] = pairs

    return results


def _emit_kv_array(name: str, pairs: list[tuple[str, str]]) -> list[str]:
    """Emit `export const NAME: DropdownOption[] = [...];` with Prettier-safe wrapping."""
    lines = [f"export const {name}: DropdownOption[] = ["]
    for label, value in pairs:
        lines.append(f"  {{ label: {_ts_string(label)}, value: {_ts_string(value)} }},")
    lines.append("];")
    return lines


def generate_dropdown_sources_ts() -> str:
    """Emit `dropdownSources.ts` — every non-schema dropdown option list.

    Sources:
    - `modules/ui/ConvertModelUI.py`: Convert-model modal dropdowns (CTk `options_kv`).
    - `modules/ui/ConceptWindow.py`: Concept modal dropdowns (CTk `options_kv`).
    - `web/scripts/ui_metadata.TOOL_DROPDOWN_OPTIONS`: Caption/Mask tool modals
      (web-specific; backend VALID_*_MODES are unordered `set`s, so ordering is
      curated here rather than AST-parsed).
    """
    global _incomplete_generation

    from web.scripts.ui_metadata import TOOL_DROPDOWN_OPTIONS

    enum_lookup = dict(collect_enums())

    convert_path = os.path.join(PROJECT_ROOT, "modules", "ui", "ConvertModelUI.py")
    concept_path = os.path.join(PROJECT_ROOT, "modules", "ui", "ConceptWindow.py")

    try:
        convert_calls = _extract_options_kv_calls(convert_path, enum_lookup)
    except OSError as e:
        print(f"  WARNING: Could not read {convert_path} ({e})")
        convert_calls = {}
        _incomplete_generation = True

    try:
        concept_calls = _extract_options_kv_calls(concept_path, enum_lookup)
    except OSError as e:
        print(f"  WARNING: Could not read {concept_path} ({e})")
        concept_calls = {}
        _incomplete_generation = True

    # Field -> exported constant name. Order here controls file layout.
    convert_map = [
        ("model_type", "CONVERT_MODEL_TYPES"),
        ("training_method", "CONVERT_TRAINING_METHODS"),
        ("output_dtype", "CONVERT_OUTPUT_DTYPES"),
        ("output_model_format", "CONVERT_OUTPUT_FORMATS"),
    ]
    concept_map = [
        ("prompt_source", "PROMPT_SOURCES"),
        ("tag_dropout_mode", "TAG_DROPOUT_MODES"),
        ("tag_dropout_special_tags_mode", "TAG_DROPOUT_SPECIAL_TAGS_MODES"),
    ]

    lines = [
        "// Auto-generated by web/scripts/generate_types.py",
        "// Do not edit manually. Regenerate when CTk options_kv calls or",
        "// ui_metadata.TOOL_DROPDOWN_OPTIONS change.",
        "",
        "/** Label/value pair consumed by `<Select options={...}>` (matches SelectKVOption). */",
        "export interface DropdownOption {",
        "  label: string;",
        "  value: string;",
        "}",
        "",
        "// --- Convert Model modal (modules/ui/ConvertModelUI.py) ----------------------",
        "",
    ]

    for field, const_name in convert_map:
        pairs = convert_calls.get(field)
        if not pairs:
            print(f"  WARNING: No options_kv for {field!r} in ConvertModelUI.py")
            _incomplete_generation = True
            pairs = []
        lines.extend(_emit_kv_array(const_name, pairs))
        lines.append("")

    lines.append("// --- Concept modal (modules/ui/ConceptWindow.py) -----------------------------")
    lines.append("")

    for field, const_name in concept_map:
        pairs = concept_calls.get(field)
        if not pairs:
            print(f"  WARNING: No options_kv for {field!r} in ConceptWindow.py")
            _incomplete_generation = True
            pairs = []
        lines.extend(_emit_kv_array(const_name, pairs))
        lines.append("")

    lines.append("// --- Caption / Mask tool modals (web/scripts/ui_metadata.py) -----------------")
    lines.append("")

    for const_name in ("CAPTION_MODELS", "CAPTION_MODES", "MASK_MODELS", "MASK_MODES"):
        pairs = TOOL_DROPDOWN_OPTIONS.get(const_name, [])
        if not pairs:
            print(f"  WARNING: TOOL_DROPDOWN_OPTIONS missing {const_name!r}")
            _incomplete_generation = True
        lines.extend(_emit_kv_array(const_name, pairs))
        lines.append("")

    return "\n".join(lines)


def write_file(filename: str, content: str) -> str:
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return filepath


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Collecting enums...")
    enums = collect_enums()
    print(f"  Found {len(enums)} enum types")

    print("Collecting config classes...")
    configs = collect_configs()
    print(f"  Found {len(configs)} config classes")

    enum_names = {name for name, _ in enums}

    print("Generating enums.ts...")
    print(f"  Wrote {write_file('enums.ts', generate_enums_ts(enums))}")

    print("Generating config.ts...")
    print(f"  Wrote {write_file('config.ts', generate_config_ts(configs, enum_names))}")

    print("Generating metadata.ts...")
    print(f"  Wrote {write_file('metadata.ts', generate_metadata_ts(configs, enum_names))}")

    new_generators = [
        ("modelTypeInfo.ts", lambda: generate_model_type_info_ts(enums)),
        ("optimizerInfo.ts", lambda: generate_optimizer_info_ts(enums)),
        ("optimizerKeyDetails.ts", lambda: generate_optimizer_key_details_ts()),
        ("enumLabels.ts", lambda: generate_enum_labels_ts(enums)),
        ("dataTypeSubsets.ts", lambda: generate_data_type_subsets_ts()),
        ("tooltips.ts", lambda: generate_tooltips_ts(configs)),
        ("layerPresets.ts", lambda: generate_layer_presets_ts()),
        ("lossWeightInfo.ts", lambda: generate_loss_weight_info_ts()),
        ("dropdownSources.ts", lambda: generate_dropdown_sources_ts()),
    ]

    global _incomplete_generation
    for filename, generator in new_generators:
        print(f"Generating {filename}...")
        try:
            print(f"  Wrote {write_file(filename, generator())}")
        except Exception as e:
            print(f"  ERROR generating {filename}: {e}")
            import traceback
            traceback.print_exc()
            _incomplete_generation = True

    backend_generated_dir = os.path.join(PROJECT_ROOT, "web", "backend", "generated")
    os.makedirs(backend_generated_dir, exist_ok=True)

    print("Generating optimizer_defaults.json (backend)...")
    try:
        json_content = generate_optimizer_defaults_json(enums)
        json_path = os.path.join(backend_generated_dir, "optimizer_defaults.json")
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_content)
        print(f"  Wrote {json_path}")
    except Exception as e:
        print(f"  ERROR generating optimizer_defaults.json: {e}")
        import traceback
        traceback.print_exc()
        _incomplete_generation = True

    print("Generating optimizer_key_details.json (backend)...")
    try:
        json_content = generate_optimizer_key_details_json()
        json_path = os.path.join(backend_generated_dir, "optimizer_key_details.json")
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json_content)
        print(f"  Wrote {json_path}")
    except Exception as e:
        print(f"  ERROR generating optimizer_key_details.json: {e}")
        import traceback
        traceback.print_exc()
        _incomplete_generation = True

    total_enum_values = sum(len(list(cls)) for _, cls in enums)
    total_config_fields = 0
    for _class_name, cls in configs:
        instance = cls.default_values()
        total_config_fields += len(instance.types)

    print("\nSummary:")
    print(f"  {len(enums)} enum types with {total_enum_values} total values")
    print(f"  {len(configs)} config interfaces with {total_config_fields} total fields")
    print("  12 generated TypeScript files")

    if _incomplete_generation:
        print("\nWARNING: Some files were not generated successfully.")
        print("Check the errors above and fix the underlying issues.")
        print("Re-run from the OneTrainer venv: python -m web.scripts.generate_types")
        sys.exit(1)
    else:
        print("Done!")


if __name__ == "__main__":
    main()
