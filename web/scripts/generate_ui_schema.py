from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.scripts import ui_metadata
from web.scripts.ui_parser import ADV_COMMAND_MAP, CtkTabParser, ParsedField, ParsedTab

logger = logging.getLogger(__name__)


def extract_predicates() -> dict[str, list[str]]:
    try:
        from modules.util.enum.ModelType import ModelType
    except ImportError:
        logger.warning("Could not import ModelType; predicates will be empty")
        return {}

    predicates: dict[str, list[str]] = {}
    for attr_name in ModelType.__dict__:
        if attr_name.startswith(("is_", "has_")):
            method = getattr(ModelType, attr_name, None)
            if callable(method):
                group = []
                for member in ModelType:
                    try:
                        if method(member):
                            group.append(member.name)
                    except Exception:  # noqa: PERF203
                        pass
                if group:
                    predicates[attr_name] = group
    return predicates


def extract_enums() -> dict[str, dict]:
    import importlib

    enum_dir = PROJECT_ROOT / "modules" / "util" / "enum"
    enums: dict[str, dict] = {}
    labels_map = getattr(ui_metadata, "ENUM_DISPLAY_LABELS", {})

    for py_file in sorted(enum_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"modules.util.enum.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and issubclass(obj, __import__("enum").Enum) and obj.__module__ == mod.__name__:
                values = [m.name for m in obj]
                curated = labels_map.get(attr_name, {})
                labels = {}
                for v in values:
                    if v in curated:
                        labels[v] = curated[v]
                    else:
                        labels[v] = _auto_label(v)
                enums[attr_name] = {"values": values, "labels": labels}

    return enums


def extract_training_methods_by_model() -> dict[str, list[str]]:
    try:
        from modules.util.enum.ModelType import ModelType
        from modules.util.enum.TrainingMethod import TrainingMethod
    except ImportError:
        return {}

    result: dict[str, list[str]] = {}
    for mt in ModelType:
        result[mt.name] = [tm.name for tm in TrainingMethod]
    return result


def extract_dtype_subsets() -> dict[str, list[dict[str, str]]]:
    raw = getattr(ui_metadata, "DTYPE_SUBSETS", {})
    result: dict[str, list[dict[str, str]]] = {}
    for key, entries in raw.items():
        result[key] = [{"label": label, "value": value} for label, value in entries]
    return result


def _sanitize_default(value):
    # Browsers' JSON.parse rejects raw Infinity / -Infinity / NaN tokens,
    # so encode them as strings the same way the backend JSON does.
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        if value != value:  # NaN
            return "NaN"
        return value
    if hasattr(value, "name"):
        return value.name
    return value


def extract_optimizer_defaults() -> dict[str, dict]:
    try:
        from web.scripts.generate_types import _extract_optimizer_defaults

        raw = _extract_optimizer_defaults()
        return {
            (k.name if hasattr(k, "name") else str(k)): {pk: _sanitize_default(pv) for pk, pv in v.items()}
            for k, v in raw.items()
        }
    except Exception:
        json_path = PROJECT_ROOT / "web" / "backend" / "generated" / "optimizer_defaults.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        logger.warning("Could not extract optimizer defaults")
        return {}


def extract_optimizer_key_details() -> dict[str, dict]:
    try:
        from web.scripts.generate_types import _extract_key_detail_map

        return _extract_key_detail_map()
    except Exception:
        json_path = PROJECT_ROOT / "web" / "backend" / "generated" / "optimizer_key_details.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        logger.warning("Could not extract optimizer key details")
        return {}


def parse_all_tabs() -> dict[str, ParsedTab]:
    parser = CtkTabParser()
    ui_dir = PROJECT_ROOT / "modules" / "ui"
    tabs: dict[str, ParsedTab] = {}

    custom_tab_ids = {"concept", "concepts", "sampling", "additionalembeddings"}

    for pattern in ["*Tab.py"]:
        for filepath in sorted(ui_dir.glob(pattern)):
            if filepath.name.startswith("_"):
                continue
            result = parser.parse_file(filepath)
            if result and result.tab_id not in custom_tab_ids:
                tabs[result.tab_id] = result
                logger.info("Parsed %s -> tab '%s' (%d sections, %d variants)",
                           filepath.name, result.tab_id,
                           len(result.sections), len(result.variants))

    train_ui_path = ui_dir / "TrainUI.py"
    if train_ui_path.exists():
        inline_parser = CtkTabParser()
        inline_parser.parse_file(train_ui_path)

        inline_tab_methods = {
            "general": "create_general_tab",
            "data": "create_data_tab",
            "backup": "create_backup_tab",
            "embedding": "embedding_tab",
        }

        for tab_id, method_name in inline_tab_methods.items():
            result = inline_parser.parse_method_as_tab(method_name, tab_id)
            if not result or not result.sections:
                continue

            existing = tabs.get(tab_id)
            if existing and existing.sections:
                merged_fields = 0
                for sec_id, sec in result.sections.items():
                    if sec_id in existing.sections:
                        existing_keys = {f.key for f in existing.sections[sec_id].fields}
                        new_fields = [f for f in sec.fields if f.key not in existing_keys]
                        existing.sections[sec_id].fields = new_fields + existing.sections[sec_id].fields
                        merged_fields += len(new_fields)
                    else:
                        existing.sections[sec_id] = sec
                        merged_fields += len(sec.fields)
                logger.info("Merged TrainUI.%s() into tab '%s' (+%d fields)",
                           method_name, tab_id, merged_fields)
            else:
                tabs[result.tab_id] = result
                logger.info("Parsed TrainUI.%s() -> tab '%s' (%d sections)",
                           method_name, result.tab_id, len(result.sections))

    return tabs


def field_to_dict(field: ParsedField) -> dict:
    d: dict = {
        "key": field.key,
        "label": field.label or _auto_label(field.key),
        "type": field.widget_type,
    }

    if field.tooltip:
        d["tooltip"] = field.tooltip
    if field.widget_type == "entry" and field.key:
        d["inputType"] = "text"
    if field.nullable:
        d["nullable"] = True
    if field.enum_ref:
        d["enumRef"] = field.enum_ref
    if field.kv_options:
        d["options"] = field.kv_options
    if field.dtype_subset:
        d["dtypeSubset"] = field.dtype_subset
    if field.adv_command:
        modal = ADV_COMMAND_MAP.get(field.adv_command)
        if modal:
            d["modal"] = modal
    if field.widget_type == "time-entry":
        d["valuePath"] = field.key
        d["unitPath"] = field.unit_var or f"{field.key}_unit"
    if field.widget_type == "layer-filter":
        d["filterPath"] = field.key
        d["presetPath"] = field.preset_var or f"{field.key}_preset"
        d["regexPath"] = field.regex_var or f"{field.key}_regex"

    return d


def _derive_sec_ref(method_name: str) -> str:
    """Strips prefixes/suffixes and handles name-mangled forms."""
    sec_ref = method_name
    for prefix in ("__create_", "_create_"):
        if sec_ref.startswith(prefix):
            sec_ref = sec_ref[len(prefix):]
    for suffix in ("_frame", "_components"):
        if sec_ref.endswith(suffix):
            sec_ref = sec_ref[:-len(suffix)]
            break
    mangled_prefix_match = re.match(r"_\w+(__create_)", sec_ref)
    if mangled_prefix_match:
        sec_ref = sec_ref[mangled_prefix_match.end(1) - len("__create_"):]
        for prefix in ("__create_", "_create_"):
            if sec_ref.startswith(prefix):
                sec_ref = sec_ref[len(prefix):]
        for suffix in ("_frame", "_components"):
            if sec_ref.endswith(suffix):
                sec_ref = sec_ref[:-len(suffix)]
                break
    return sec_ref


def _expand_field_template(field_dict: dict, i: int) -> dict:
    """Resolve ``{...}`` placeholders: suffix for config keys, number for labels."""
    suffix = f"_{i}" if i > 1 else ""
    number = str(i)
    out = dict(field_dict)

    for path_key in ("key", "valuePath", "unitPath", "filterPath", "presetPath", "regexPath"):
        if path_key in out and "{...}" in out[path_key]:
            out[path_key] = out[path_key].replace("{...}", suffix)

    for text_key in ("label", "tooltip"):
        if text_key in out and isinstance(out[text_key], str) and "{...}" in out[text_key]:
            out[text_key] = out[text_key].replace("{...}", number)

    return out


def _expand_section_template(
    base_section_dict: dict,
    i: int,
    base_section_id: str,
) -> tuple[str, dict]:
    core_id = base_section_id
    qualifier = ""
    dunder_pos = base_section_id.find("__")
    if dunder_pos >= 0:
        core_id = base_section_id[:dunder_pos]
        qualifier = base_section_id[dunder_pos:]

    base_name = core_id[:-2] if core_id.endswith("_n") else core_id
    expanded_id = f"{base_name}_{i}{qualifier}"
    expanded = dict(base_section_dict)
    expanded["id"] = expanded_id
    expanded["label"] = f"Text Encoder {i}"
    expanded["fields"] = [_expand_field_template(f, i) for f in base_section_dict["fields"]]
    return expanded_id, expanded


def _expand_templated_sections(
    section_defs: dict[str, dict],
    variants: list,
    tab_variants: list,
):
    template_ids: set[str] = set()
    for sec_id, sec_dict in section_defs.items():
        for field in sec_dict.get("fields", []):
            if "{...}" in field.get("key", ""):
                template_ids.add(sec_id)
                break

    if not template_ids:
        return

    for var_idx, parsed_variant in enumerate(tab_variants):
        if var_idx >= len(variants):
            break
        var_dict = variants[var_idx]

        fc_by_sec: list[tuple[str, str, object]] = []
        for fc in parsed_variant.frame_calls:
            sec_ref = fc.resolved_section_id or _derive_sec_ref(fc.method_name)
            fc_by_sec.append((fc.column_var, sec_ref, fc))

        col_map: dict[str, list[str]] = {}
        for col_var, sec_ref, fc in fc_by_sec:
            if col_var not in col_map:
                col_map[col_var] = []

            if sec_ref in template_ids and "i" in fc.kwargs:
                i_val = fc.kwargs["i"]
                core_ref = sec_ref
                qualifier = ""
                dunder_pos = sec_ref.find("__")
                if dunder_pos >= 0:
                    core_ref = sec_ref[:dunder_pos]
                    qualifier = sec_ref[dunder_pos:]
                base_name = core_ref[:-2] if core_ref.endswith("_n") else core_ref
                expanded_id = f"{base_name}_{i_val}{qualifier}"
                if expanded_id not in section_defs:
                    _, expanded_dict = _expand_section_template(
                        section_defs[sec_ref], i_val, sec_ref
                    )
                    expanded_dict["id"] = expanded_id
                    section_defs[expanded_id] = expanded_dict
                col_map[col_var].append(expanded_id)
            else:
                col_map[col_var].append(sec_ref)

        sorted_cols = sorted(col_map.keys())
        var_dict["columns"] = [{"sections": col_map[c]} for c in sorted_cols]

    for tmpl_id in template_ids:
        still_referenced = False
        for var_dict in variants:
            for col in var_dict.get("columns", []):
                if tmpl_id in col.get("sections", []):
                    still_referenced = True
                    break
            if still_referenced:
                break
        if not still_referenced and tmpl_id in section_defs:
            del section_defs[tmpl_id]


def tab_to_dict(tab: ParsedTab) -> dict:
    d: dict = {
        "id": tab.tab_id,
        "label": TAB_LABEL_OVERRIDES.get(tab.tab_id, _auto_label(tab.tab_id)),
        "renderer": "schema",
    }

    if tab.variants:
        d["layout"] = tab.layout

        section_defs: dict[str, dict] = {}
        for section_id, section in tab.sections.items():
            section_defs[section_id] = {
                "id": section.id,
                "label": section.label,
                "fields": [field_to_dict(f) for f in section.fields],
            }

        variants = []
        for variant in tab.variants:
            columns: dict[str, list[str]] = {}
            for fc in variant.frame_calls:
                col = fc.column_var
                if col not in columns:
                    columns[col] = []
                sec_ref = fc.resolved_section_id or _derive_sec_ref(fc.method_name)
                columns[col].append(sec_ref)

            sorted_cols = sorted(columns.keys())
            col_defs = [{"sections": columns[c]} for c in sorted_cols]

            variants.append({
                "when": {"predicate": variant.predicate},
                "columns": col_defs,
            })

        _expand_templated_sections(section_defs, variants, tab.variants)

        d["sectionDefs"] = section_defs
        d["variants"] = variants
    else:
        d["layout"] = "single-column"
        d["sections"] = [
            {
                "id": section.id,
                "label": section.label,
                "fields": [field_to_dict(f) for f in section.fields],
            }
            for section in tab.sections.values()
        ]

    return d


def enrich_with_tooltips(schema: dict):
    tooltips = getattr(ui_metadata, "FIELD_TOOLTIPS", {})
    if not tooltips:
        return

    for tab in schema.get("tabs", []):
        for section in _iter_all_sections(tab):
            for field in section.get("fields", []):
                key = field.get("key", "")
                if key in tooltips and not field.get("tooltip"):
                    field["tooltip"] = tooltips[key]


def enrich_field_visibility(schema: dict):
    FIELD_VISIBILITY_OVERRIDES: dict[tuple[str, str], dict] = {
        ("loss", "vb_loss_strength"): {"predicate": "is_pixart"},
        ("base2", "frames"): {
            "or": [
                {"predicate": "is_hunyuan_video"},
                {"predicate": "is_hi_dream"},
            ]
        },
        ("noise", "generalized_offset_noise"): {
            "or": [
                {"predicate": "is_stable_diffusion"},
                {"predicate": "is_stable_diffusion_xl"},
            ]
        },
        ("noise", "dynamic_timestep_shifting"): {
            "or": [
                {"predicate": "is_flux_1"},
                {"predicate": "is_flux_2"},
                {"predicate": "is_qwen"},
                {"predicate": "is_z_image"},
            ]
        },
        ("loss", "loss_weight_strength"): {"not": {"predicate": "is_flow_matching"}},
    }

    for tab in schema.get("tabs", []):
        for section in _iter_all_sections(tab):
            sec_id = section.get("id", "")
            for field in section.get("fields", []):
                key = field.get("key", "")
                vis = FIELD_VISIBILITY_OVERRIDES.get((sec_id, key))
                if vis is not None:
                    field["visibility"] = vis


def enrich_field_types(schema: dict):
    try:
        from modules.util.config.TrainConfig import TrainConfig
        config = TrainConfig.default_values()
        type_map = {}
        _collect_types(config, "", type_map)
    except Exception:
        return

    for tab in schema.get("tabs", []):
        for section in _iter_all_sections(tab):
            for field in section.get("fields", []):
                key = field.get("key", "")
                if field.get("type") == "entry" and key in type_map:
                    if type_map[key] in (int, float):
                        field["inputType"] = "number"


def _collect_types(config, prefix: str, type_map: dict):
    if not hasattr(config, "types"):
        return
    for name, typ in config.types.items():
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        type_map[path] = typ
        val = getattr(config, name, None)
        if val is not None and hasattr(val, "types"):
            _collect_types(val, f"{path}.", type_map)


def _iter_all_sections(tab: dict):
    if "sections" in tab:
        yield from tab["sections"]
    if "sectionDefs" in tab:
        yield from tab["sectionDefs"].values()


TAB_LABEL_OVERRIDES = {
    "additionalembeddings": "Additional Embeddings",
    "lora": "LoRA",
}

CUSTOM_TABS = [
    {"id": "concepts", "label": "Concepts", "renderer": "custom"},
    {"id": "sampling", "label": "Sampling", "renderer": "custom"},
    {"id": "additionalembeddings", "label": "Additional Embeddings", "renderer": "custom"},
    {"id": "performance", "label": "Performance", "renderer": "custom"},
    {"id": "run", "label": "Run", "renderer": "custom"},
    {"id": "tools", "label": "Tools", "renderer": "custom"},
    {"id": "help", "label": "Help", "renderer": "custom"},
]

TAB_ORDER = [
    "general", "model", "data", "concepts", "training", "sampling",
    "backup", "tools", "lora", "embedding",
    "additionalembeddings", "cloud", "performance", "run", "help",
]


def build_modals() -> dict:
    return {
        "optimizer-params": {
            "title": "Optimizer Parameters",
            "renderer": "optimizer-params",
        },
        "scheduler-params": {
            "title": "Scheduler Parameters",
            "renderer": "scheduler-params",
        },
        "timestep-dist": {
            "title": "Timestep Distribution",
            "renderer": "timestep-dist",
        },
        "offloading": {
            "title": "Offloading & Checkpointing",
            "renderer": "offloading",
        },
    }


def generate_schema() -> dict:
    parsed_tabs = parse_all_tabs()

    tab_dicts: dict[str, dict] = {}
    for tab_id, parsed in parsed_tabs.items():
        tab_dicts[tab_id] = tab_to_dict(parsed)

    for custom in CUSTOM_TABS:
        tab_dicts[custom["id"]] = custom

    if "lora" in tab_dicts:
        tab_dicts["lora"]["visibility"] = {"field": "training_method", "eq": "LORA"}
        for section in tab_dicts["lora"].get("sections", []):
            sid = section["id"]
            if sid == "lora_lora":
                section["visibility"] = {"field": "peft_type", "eq": "LORA"}
            elif sid == "lora_lora_or_loha":
                section["visibility"] = {"field": "peft_type", "in": ["LORA", "LOHA"]}
            elif sid == "lora_oft_2":
                section["visibility"] = {"field": "peft_type", "eq": "OFT_2"}
    if "embedding" in tab_dicts:
        tab_dicts["embedding"]["visibility"] = {"field": "training_method", "eq": "EMBEDDING"}
    if "additionalembeddings" in tab_dicts:
        tab_dicts["additionalembeddings"]["visibility"] = {"not": {"predicate": "is_flow_matching"}}

    ordered_tabs = [tab_dicts[tab_id] for tab_id in TAB_ORDER if tab_id in tab_dicts]
    ordered_tabs.extend(tab for tab_id, tab in tab_dicts.items() if tab_id not in TAB_ORDER)

    predicates = extract_predicates()
    enums = extract_enums()
    dtype_subsets = extract_dtype_subsets()

    schema = {
        "$version": 1,
        "predicates": predicates,
        "enums": enums,
        "dtypeSubsets": dtype_subsets,
        "tabs": ordered_tabs,
        "modals": build_modals(),
        "optimizerDefaults": extract_optimizer_defaults(),
        "optimizerKeyDetails": extract_optimizer_key_details(),
        "trainingMethodsByModel": extract_training_methods_by_model(),
    }

    enrich_with_tooltips(schema)
    enrich_field_types(schema)
    enrich_field_visibility(schema)

    return schema


def write_schema(schema: dict):
    gui_dir = PROJECT_ROOT / "web" / "gui"

    public_path = gui_dir / "public" / "ui-schema.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    with open(public_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", public_path)

    dist_path = gui_dir / "dist" / "renderer" / "ui-schema.json"
    if dist_path.parent.exists():
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s", dist_path)


def _auto_label(name: str) -> str:
    name = name.split(".")[-1]
    words = name.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    validate = "--validate" in sys.argv

    schema = generate_schema()

    if validate:
        tab_count = len(schema["tabs"])
        field_count = sum(
            len(f.get("fields", []))
            for tab in schema["tabs"]
            for section in _iter_all_sections(tab)
            for f in [section]
        )
        print(f"Schema: {tab_count} tabs, {field_count} fields")
        print(f"Predicates: {len(schema['predicates'])}")
        print(f"Enums: {len(schema['enums'])}")
        return

    write_schema(schema)
    print("UI schema generated successfully.")


if __name__ == "__main__":
    main()
