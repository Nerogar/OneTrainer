from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedLabel:
    frame_var: str
    row: int
    col: int
    text: str
    tooltip: str | None = None


@dataclass
class ParsedField:
    frame_var: str
    row: int
    col: int
    key: str
    widget_type: str
    tooltip: str | None = None
    label: str | None = None
    enum_ref: str | None = None
    kv_options: list[dict[str, str]] | None = None
    path_mode: str | None = None
    unit_var: str | None = None
    adv_command: str | None = None
    preset_var: str | None = None
    regex_var: str | None = None
    dtype_subset: str | None = None
    required: bool = False
    nullable: bool = False


@dataclass
class ParsedSection:
    id: str
    label: str
    fields: list[ParsedField] = dataclass_field(default_factory=list)
    param_flags: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class ParsedFrameCall:
    method_name: str
    column_var: str
    kwargs: dict[str, Any] = dataclass_field(default_factory=dict)
    resolved_section_id: str | None = None


@dataclass
class ParsedVariant:
    predicate: str
    setup_method: str
    frame_calls: list[ParsedFrameCall] = dataclass_field(default_factory=list)


@dataclass
class ParsedTab:
    tab_id: str
    class_name: str
    layout: str
    variants: list[ParsedVariant] = dataclass_field(default_factory=list)
    sections: dict[str, ParsedSection] = dataclass_field(default_factory=dict)


WIDGET_TYPE_MAP = {
    "entry": "entry",
    "switch": "toggle",
    "options": "select",
    "options_adv": "select-adv",
    "options_kv": "select-kv",
    "path_entry": "file",
    "time_entry": "time-entry",
    "layer_filter_entry": "layer-filter",
}

ADV_COMMAND_MAP = {
    "__open_optimizer_params_window": "optimizer-params",
    "_TrainingTab__open_optimizer_params_window": "optimizer-params",
    "__open_scheduler_params_window": "scheduler-params",
    "_TrainingTab__open_scheduler_params_window": "scheduler-params",
    "__open_offloading_window": "offloading",
    "_TrainingTab__open_offloading_window": "offloading",
    "__open_timestep_distribution_window": "timestep-dist",
    "_TrainingTab__open_timestep_distribution_window": "timestep-dist",
}


def _get_str(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append("{...}")
        return "".join(parts)
    return None


def _get_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _get_int(node.operand)
        return -v if v is not None else None
    return None


def _get_int_or_lineno(node: ast.expr, call_node: ast.AST) -> int:
    val = _get_int(node)
    if val is not None:
        return val
    # Use negative line number to avoid collisions with real row numbers
    return -(getattr(call_node, "lineno", 0))


def _get_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _get_kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _get_kwarg_str(call: ast.Call, name: str) -> str | None:
    node = _get_kwarg(call, name)
    return _get_str(node) if node else None


def _get_kwarg_bool(call: ast.Call, name: str) -> bool | None:
    node = _get_kwarg(call, name)
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _extract_enum_ref(node: ast.expr) -> str | None:
    if isinstance(node, ast.ListComp):
        for gen in node.generators:
            iter_node = gen.iter
            if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name):
                if iter_node.func.id == "list" and len(iter_node.args) == 1:
                    return _get_name(iter_node.args[0])
    return None


def _extract_kv_options(
    node: ast.expr,
    local_lists: dict[str, ast.List] | None = None,
) -> list[dict[str, str]] | None:
    # Resolve `options_kv(..., my_var, ...)` where `my_var = [(label, Enum.X), ...]`
    # is assigned at the top of the enclosing method (e.g. RLHFTab.refresh_ui).
    if isinstance(node, ast.Name) and local_lists:
        resolved = local_lists.get(node.id)
        if resolved is None:
            return None
        node = resolved
    if not isinstance(node, ast.List):
        return None
    options = []
    for elt in node.elts:
        if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
            label = _get_str(elt.elts[0])
            value = _get_name(elt.elts[1])
            if label and value:
                if "." in value:
                    value = value.split(".")[-1]
                options.append({"label": label, "value": value})
    return options if options else None


def _extract_dtype_subset(node: ast.expr) -> str | None:
    """Detect calls like ``self.__create_dtype_options(include_gguf=?, include_a8=?)``
    and map them to a named subset in ``DTYPE_SUBSETS``.
    """
    if not isinstance(node, ast.Call):
        return None
    func_name = _get_name(node.func)
    if not func_name:
        return None
    # Accept both ``self.__create_dtype_options`` and the name-mangled form.
    if not func_name.endswith("__create_dtype_options") and not func_name.endswith("_create_dtype_options"):
        return None

    include_gguf = _get_kwarg_bool(node, "include_gguf") or False
    include_a8 = _get_kwarg_bool(node, "include_a8") or False

    if include_gguf:
        return "with_gguf_a8"
    if include_a8:
        return "with_a8"
    return "base"


def _extract_adv_command_name(call: ast.Call) -> str | None:
    node = _get_kwarg(call, "adv_command")
    if node is None:
        return None
    name = _get_name(node)
    if name and name.startswith("self."):
        return name[5:]
    return name


class CtkTabParser:

    def __init__(self):
        self._tree: ast.Module | None = None
        self._source: str = ""
        self._class_node: ast.ClassDef | None = None
        # Populated per section: top-level `name = [...]` assignments in the
        # method being walked, so options_kv calls that reference a local list
        # variable (e.g. RLHFTab.refresh_ui) can resolve it.
        self._local_lists: dict[str, ast.List] = {}

    @staticmethod
    def _collect_local_list_assignments(stmts: list[ast.stmt]) -> dict[str, ast.List]:
        result: dict[str, ast.List] = {}
        for stmt in stmts:
            if not isinstance(stmt, ast.Assign):
                continue
            if not isinstance(stmt.value, ast.List):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                result[target.id] = stmt.value
        return result

    def parse_file(self, filepath: str | Path) -> ParsedTab | None:
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning("File not found: %s", filepath)
            return None

        self._source = filepath.read_text(encoding="utf-8")
        try:
            self._tree = ast.parse(self._source)
        except SyntaxError:
            logger.error("Syntax error parsing %s", filepath)
            return None

        self._class_node = self._find_class()
        if not self._class_node:
            logger.warning("No class found in %s", filepath)
            return None

        class_name = self._class_node.name
        tab_id = self._derive_tab_id(class_name, filepath.stem)

        layout, variants = self._parse_refresh_ui()

        for variant in variants:
            variant.frame_calls = self._parse_setup_method(variant.setup_method)

        sections = self._parse_sections_from_variants(variants)

        if not variants and not sections:
            sections = self._parse_static_tab(tab_id)

        return ParsedTab(
            tab_id=tab_id,
            class_name=class_name,
            layout=layout,
            variants=variants,
            sections=sections,
        )

    def parse_method_as_tab(self, method_name: str, tab_id: str) -> ParsedTab | None:
        if not self._class_node:
            return None

        method = self._find_method(method_name)
        if not method:
            logger.warning("Method %s not found in %s", method_name, self._class_node.name)
            return None

        section = self._extract_section_from_body(method.body, tab_id, method)
        if not section or not section.fields:
            return None

        return ParsedTab(
            tab_id=tab_id,
            class_name=self._class_node.name,
            layout="single-column",
            variants=[],
            sections={section.id: section},
        )

    def _find_class(self) -> ast.ClassDef | None:
        if not self._tree:
            return None
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef):
                return node
        return None

    def _derive_tab_id(self, class_name: str, file_stem: str) -> str:
        name = class_name
        for suffix in ("Tab", "Window", "UI"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        return name.lower()

    def _find_method(self, name: str) -> ast.FunctionDef | None:
        if not self._class_node:
            return None
        mangled = f"_{self._class_node.name}{name}" if name.startswith("__") else name
        for node in self._class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == name or node.name == mangled:
                    return node
        return None

    def _parse_refresh_ui(self) -> tuple[str, list[ParsedVariant]]:
        method = self._find_method("refresh_ui")
        if not method:
            return "single-column", []

        column_vars = set()
        for node in ast.walk(method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = _get_name(target)
                    if name and re.match(r"column_\d+", name):
                        column_vars.add(name)

        layout = "three-column" if len(column_vars) >= 3 else "single-column"

        # Only iterate direct body statements (not ast.walk) to avoid duplicates
        variants: list[ParsedVariant] = []
        for node in method.body:
            if isinstance(node, ast.If):
                self._extract_variants_from_if(node, variants)

        return layout, variants

    def _extract_variants_from_if(self, if_node: ast.If, variants: list[ParsedVariant]):
        predicate = self._extract_model_type_predicate(if_node.test)
        if predicate:
            setup_method = self._extract_setup_call(if_node.body)
            if setup_method:
                variants.append(ParsedVariant(
                    predicate=predicate,
                    setup_method=setup_method,
                ))

        for elif_node in if_node.orelse:
            if isinstance(elif_node, ast.If):
                self._extract_variants_from_if(elif_node, variants)

    def _extract_model_type_predicate(self, test: ast.expr) -> str | None:
        if isinstance(test, ast.Call):
            func_name = _get_name(test.func)
            if func_name:
                match = re.search(r"model_type\.(is_\w+|has_\w+)", func_name)
                if match:
                    return match.group(1)
        return None

    def _extract_setup_call(self, body: list[ast.stmt]) -> str | None:
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func_name = _get_name(stmt.value.func)
                if func_name and "setup" in func_name.lower():
                    if func_name.startswith("self."):
                        return func_name[5:]
                    return func_name
        return None

    def _parse_setup_method(self, method_name: str) -> list[ParsedFrameCall]:
        method = self._find_method(method_name)
        if not method:
            return []

        frame_calls: list[ParsedFrameCall] = []
        for stmt in method.body:
            call: ast.Call | None = None
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) or isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call = stmt.value

            if call is None:
                continue

            func_name = _get_name(call.func)
            if not func_name:
                continue

            if not (func_name.startswith("self.") and "create" in func_name.lower()):
                continue

            clean_name = func_name[5:]

            col_var = "column_0"
            if len(call.args) >= 1:
                col_var = _get_name(call.args[0]) or "column_0"

            kwargs: dict[str, Any] = {}
            for kw in call.keywords:
                if kw.arg:
                    val = _get_kwarg_bool(call, kw.arg)
                    if val is None:
                        int_val = _get_int(kw.value)
                        val = int_val if int_val is not None else _get_str(kw.value)
                    kwargs[kw.arg] = val

            if "i" not in kwargs:
                for kw in call.keywords:
                    if kw.arg == "i":
                        val = _get_int(kw.value)
                        if val is not None:
                            kwargs["i"] = val

            frame_calls.append(ParsedFrameCall(
                method_name=clean_name,
                column_var=col_var,
                kwargs=kwargs,
            ))

        return frame_calls

    def _get_method_param_defaults(self, method_name: str) -> dict[str, Any]:
        method = self._find_method(method_name)
        if not method or not method.args:
            return {}

        defaults: dict[str, Any] = {}
        all_args = [a for a in method.args.args if a.arg != "self"]
        all_defaults = method.args.defaults

        # Defaults are right-aligned with args
        offset = len(all_args) - len(all_defaults)
        for i, arg in enumerate(all_args):
            if i >= offset:
                default_node = all_defaults[i - offset]
                if isinstance(default_node, ast.Constant):
                    defaults[arg.arg] = default_node.value

        return defaults

    def _parse_sections_from_variants(
        self,
        variants: list[ParsedVariant],
    ) -> dict[str, ParsedSection]:
        """Parse frame methods with per-variant kwargs to produce filtered sections.

        When different variants pass different kwargs to the same method,
        variant-specific sections are created with qualified IDs (e.g. ``base__1``).
        """
        defaults_cache: dict[str, dict[str, Any]] = {}
        combo_to_fcs: dict[tuple[str, frozenset], list[ParsedFrameCall]] = {}
        method_to_sigs: dict[str, list[frozenset]] = {}

        for variant in variants:
            for fc in variant.frame_calls:
                mn = fc.method_name
                if mn not in defaults_cache:
                    defaults_cache[mn] = self._get_method_param_defaults(mn)

                # Separate filter kwargs from the ``i`` kwarg (handled by template expansion)
                filter_kw = {k: v for k, v in fc.kwargs.items() if k != "i"}
                merged = dict(defaults_cache[mn])
                merged.update(filter_kw)
                sig = frozenset(sorted(merged.items()))

                key = (mn, sig)
                combo_to_fcs.setdefault(key, []).append(fc)
                if mn not in method_to_sigs:
                    method_to_sigs[mn] = []
                if sig not in method_to_sigs[mn]:
                    method_to_sigs[mn].append(sig)

        sections: dict[str, ParsedSection] = {}
        combo_to_section_id: dict[tuple[str, frozenset], str] = {}

        for (mn, sig) in combo_to_fcs:
            merged_kwargs = dict(sig)
            section = self._parse_frame_method(mn, call_kwargs=merged_kwargs)
            if not section:
                continue

            base_id = section.id
            sigs = method_to_sigs[mn]

            if len(sigs) > 1:
                idx = sigs.index(sig)
                if idx > 0:
                    section = ParsedSection(
                        id=f"{base_id}__{idx}",
                        label=section.label,
                        fields=list(section.fields),
                        param_flags=dict(section.param_flags),
                    )

            sections[section.id] = section
            combo_to_section_id[(mn, sig)] = section.id

        for variant in variants:
            for fc in variant.frame_calls:
                mn = fc.method_name
                filter_kw = {k: v for k, v in fc.kwargs.items() if k != "i"}
                merged = dict(defaults_cache.get(mn, {}))
                merged.update(filter_kw)
                sig = frozenset(sorted(merged.items()))
                fc.resolved_section_id = combo_to_section_id.get((mn, sig))

        return sections

    def _parse_static_tab(self, tab_id: str) -> dict[str, ParsedSection]:
        sections: dict[str, ParsedSection] = {}

        lora_method = self._find_method("setup_lora")
        if lora_method:
            refresh = self._find_method("refresh_ui")
            if refresh:
                header = self._extract_section_from_body(refresh.body, f"{tab_id}_header", refresh)
                if header and header.fields:
                    sections[header.id] = header

            lora_sections = self._parse_peft_method(lora_method, tab_id)
            sections.update(lora_sections)
            return sections

        init_method = self._find_method("__init__")
        if init_method:
            section = self._extract_section_from_body(init_method.body, tab_id, init_method)
            if section and section.fields:
                sections[section.id] = section

        if not sections:
            widget_sections = self._parse_nested_widget_class(tab_id)
            sections.update(widget_sections)

        return sections

    def _parse_nested_widget_class(self, tab_id: str) -> dict[str, ParsedSection]:
        if not self._tree or not self._class_node:
            return {}

        sections: dict[str, ParsedSection] = {}

        for node in ast.iter_child_nodes(self._tree):
            if isinstance(node, ast.ClassDef) and node is not self._class_node:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        section = self._extract_section_from_body(
                            method.body, tab_id, method
                        )
                        if section and section.fields:
                            section.label = _label_from_key(tab_id)
                            sections[section.id] = section
                            logger.info(
                                "Parsed nested widget class %s.__init__() -> %d fields",
                                node.name, len(section.fields),
                            )

        return sections

    _PEFT_DISPLAY_NAMES: dict[str, str] = {
        "LORA": "LoRA",
        "LOHA": "LoHa",
        "OFT_2": "OFT v2",
    }

    def _peft_display_name(self, branch_label: str) -> str:
        if branch_label in self._PEFT_DISPLAY_NAMES:
            return self._PEFT_DISPLAY_NAMES[branch_label]
        for part in branch_label.split("_or_"):
            if part in self._PEFT_DISPLAY_NAMES:
                return self._PEFT_DISPLAY_NAMES[part]
        return "LoRA"

    @staticmethod
    def _expand_peft_templates(section: ParsedSection, display_name: str) -> None:
        for field in section.fields:
            if field.label and "{...}" in field.label:
                field.label = field.label.replace("{...}", display_name)
            if field.tooltip and "{...}" in field.tooltip:
                field.tooltip = field.tooltip.replace("{...}", display_name)

    def _parse_peft_method(self, method: ast.FunctionDef, tab_id: str) -> dict[str, ParsedSection]:
        sections: dict[str, ParsedSection] = {}

        unconditional_stmts: list[ast.stmt] = []
        conditional_branches: list[tuple[str, list[ast.stmt]]] = []

        for stmt in method.body:
            if isinstance(stmt, ast.If):
                branch_label = self._extract_peft_condition_label(stmt.test)
                if branch_label:
                    conditional_branches.append((branch_label, stmt.body))
                    remaining = stmt.orelse
                    while remaining:
                        if len(remaining) == 1 and isinstance(remaining[0], ast.If):
                            elif_node = remaining[0]
                            elif_label = self._extract_peft_condition_label(elif_node.test)
                            if elif_label:
                                conditional_branches.append((elif_label, elif_node.body))
                            remaining = elif_node.orelse
                        else:
                            conditional_branches.append(("else", remaining))
                            break
                else:
                    unconditional_stmts.append(stmt)
            else:
                unconditional_stmts.append(stmt)

        if unconditional_stmts:
            shared = self._extract_section_from_body(
                unconditional_stmts, f"{tab_id}_shared", method
            )
            if shared and shared.fields:
                self._expand_peft_templates(shared, self._peft_display_name("shared"))
                sections[shared.id] = shared

        for branch_label, branch_body in conditional_branches:
            section_id = f"{tab_id}_{branch_label.lower()}"
            section = self._extract_section_from_body(branch_body, section_id, method)
            if section and section.fields:
                section.label = f"{_label_from_key(tab_id)} ({branch_label})"
                self._expand_peft_templates(section, self._peft_display_name(branch_label))
                sections[section.id] = section

        return sections

    def _extract_peft_condition_label(self, test: ast.expr) -> str | None:
        if isinstance(test, ast.Compare):
            for comparator in test.comparators:
                name = _get_name(comparator)
                if name and "PeftType." in name:
                    return name.split(".")[-1]
        if isinstance(test, ast.BoolOp):
            parts = []
            for val in test.values:
                label = self._extract_peft_condition_label(val)
                if label:
                    parts.append(label)
            if parts:
                return "_or_".join(parts)
        return None

    def _extract_section_from_body(
        self,
        stmts: list[ast.stmt],
        section_id: str,
        method: ast.FunctionDef,
    ) -> ParsedSection | None:
        labels: list[ParsedLabel] = []
        fields: list[ParsedField] = []
        subframe_map: dict[str, tuple[str, int, int]] = {}

        prev_locals = self._local_lists
        self._local_lists = self._collect_local_list_assignments(stmts)
        try:
            self._walk_for_components(stmts, labels, fields, "frame", method, None, subframe_map)
        finally:
            self._local_lists = prev_locals

        if not fields:
            return None

        self._correlate_labels_to_fields(labels, fields, subframe_map)

        section_label = _label_from_key(section_id)

        return ParsedSection(
            id=section_id,
            label=section_label,
            fields=fields,
        )

    def _correlate_labels_to_fields(
        self,
        labels: list[ParsedLabel],
        fields: list[ParsedField],
        subframe_map: dict[str, tuple[str, int, int]] | None = None,
    ):
        if subframe_map is None:
            subframe_map = {}

        label_map: dict[tuple[str, int, int], ParsedLabel] = {}
        label_row_map: dict[tuple[str, int], list[ParsedLabel]] = {}
        for lbl in labels:
            if lbl.row >= 0:
                label_map[(lbl.frame_var, lbl.row, lbl.col)] = lbl
                key = (lbl.frame_var, lbl.row)
                if key not in label_row_map:
                    label_row_map[key] = []
                label_row_map[key].append(lbl)

        for fld in fields:
            if fld.row >= 0:
                col_key = (fld.frame_var, fld.row, fld.col - 1)
                if col_key in label_map:
                    lbl = label_map[col_key]
                    if not fld.label:
                        fld.label = lbl.text
                    if not fld.tooltip and lbl.tooltip:
                        fld.tooltip = lbl.tooltip
                    continue

                # Sub-frame lookup: field's frame is a sub-frame placed in a parent via .grid()
                if fld.frame_var in subframe_map:
                    parent_frame, parent_row, parent_col = subframe_map[fld.frame_var]
                    parent_key = (parent_frame, parent_row, parent_col - 1)
                    if parent_key in label_map:
                        lbl = label_map[parent_key]
                        if not fld.label:
                            fld.label = lbl.text
                        if not fld.tooltip and lbl.tooltip:
                            fld.tooltip = lbl.tooltip
                        continue

                row_key = (fld.frame_var, fld.row)
                if row_key in label_row_map:
                    candidates = [lb for lb in label_row_map[row_key] if lb.col < fld.col]
                    if candidates:
                        lbl = max(candidates, key=lambda lb: lb.col)
                        if not fld.label:
                            fld.label = lbl.text
                        if not fld.tooltip and lbl.tooltip:
                            fld.tooltip = lbl.tooltip

        unmatched_fields = [f for f in fields if not f.label]
        if unmatched_fields:
            all_items: list[tuple[int, str, ParsedLabel | ParsedField]] = [
                (abs(lbl.row), "label", lbl) for lbl in labels
            ]
            all_items.extend((abs(fld.row), "field", fld) for fld in fields)
            all_items.sort(key=lambda x: x[0])

            pending_label: ParsedLabel | None = None
            for _, kind, item in all_items:
                if kind == "label":
                    pending_label = item  # type: ignore
                elif kind == "field" and pending_label and not item.label:  # type: ignore
                    fld = item  # type: ignore
                    fld.label = pending_label.text
                    if not fld.tooltip and pending_label.tooltip:
                        fld.tooltip = pending_label.tooltip
                    pending_label = None

        for fld in fields:
            if not fld.label:
                fld.label = _label_from_key(fld.key)

    def _parse_frame_method(self, method_name: str, call_kwargs: dict[str, Any] | None = None) -> ParsedSection | None:
        method = self._find_method(method_name)
        if not method:
            logger.warning("Method not found: %s", method_name)
            return None

        section_id = method_name
        for prefix in ("__create_", "_create_"):
            if section_id.startswith(prefix):
                section_id = section_id[len(prefix):]
        for suffix in ("_frame", "_components"):
            if section_id.endswith(suffix):
                section_id = section_id[:-len(suffix)]
                break

        labels: list[ParsedLabel] = []
        fields: list[ParsedField] = []
        subframe_map: dict[str, tuple[str, int, int]] = {}

        prev_locals = self._local_lists
        self._local_lists = self._collect_local_list_assignments(method.body)
        try:
            self._walk_for_components(method.body, labels, fields, "frame", method, call_kwargs, subframe_map)
        finally:
            self._local_lists = prev_locals

        self._correlate_labels_to_fields(labels, fields, subframe_map)

        section_label = _label_from_key(section_id)

        param_flags = {}
        if method.args:
            for arg in method.args.args:
                if arg.arg not in ("self", "master", "row"):
                    param_flags[arg.arg] = None
            for default, arg in zip(
                reversed(method.args.defaults),
                reversed(method.args.args),
                strict=False,
            ):
                if arg.arg in param_flags:
                    if isinstance(default, ast.Constant):
                        param_flags[arg.arg] = default.value

        return ParsedSection(
            id=section_id,
            label=section_label,
            fields=fields,
            param_flags=param_flags,
        )

    def _walk_for_components(
        self,
        stmts: list[ast.stmt],
        labels: list[ParsedLabel],
        fields: list[ParsedField],
        frame_var: str,
        method: ast.FunctionDef,
        call_kwargs: dict[str, Any] | None = None,
        subframe_map: dict[str, tuple[str, int, int]] | None = None,
    ):
        subframe_parents: dict[str, str] = {}

        for stmt in stmts:
            if isinstance(stmt, ast.If):
                cond_name = _get_name(stmt.test)
                if cond_name and call_kwargs is not None:
                    cond_val = call_kwargs.get(cond_name)
                    if cond_val is True or cond_val is None:
                        self._walk_for_components(stmt.body, labels, fields, frame_var, method, call_kwargs, subframe_map)
                    if stmt.orelse:
                        self._walk_for_components(stmt.orelse, labels, fields, frame_var, method, call_kwargs, subframe_map)
                else:
                    self._walk_for_components(stmt.body, labels, fields, frame_var, method, call_kwargs, subframe_map)
                    if stmt.orelse:
                        self._walk_for_components(stmt.orelse, labels, fields, frame_var, method, call_kwargs, subframe_map)
                continue

            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                assign_func = _get_name(stmt.value.func)
                if assign_func and assign_func.endswith("CTkFrame"):
                    if len(stmt.value.args) >= 1 and len(stmt.targets) == 1:
                        sf_var = _get_name(stmt.targets[0])
                        sf_parent = _get_name(stmt.value.args[0])
                        if sf_var and sf_parent:
                            subframe_parents[sf_var] = sf_parent

            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                grid_func = _get_name(stmt.value.func)
                if grid_func and grid_func.endswith(".grid"):
                    sf_var = grid_func.rsplit(".grid", 1)[0]
                    if sf_var in subframe_parents:
                        grid_row_node = _get_kwarg(stmt.value, "row")
                        grid_col_node = _get_kwarg(stmt.value, "column")
                        grid_row = _get_int(grid_row_node) if grid_row_node else None
                        grid_col = _get_int(grid_col_node) if grid_col_node else None
                        if grid_row is not None and grid_col is not None and subframe_map is not None:
                            subframe_map[sf_var] = (
                                subframe_parents[sf_var],
                                grid_row,
                                grid_col,
                            )

            calls = []
            if isinstance(stmt, (ast.Expr, ast.Assign)) and isinstance(stmt.value, ast.Call):
                calls.append(stmt.value)
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Tuple):
                calls.extend(elt for elt in stmt.value.elts if isinstance(elt, ast.Call))

            for call in calls:
                func_name = _get_name(call.func)
                if not func_name:
                    continue

                if func_name.startswith("components."):
                    widget = func_name.split(".")[-1]
                    if widget == "label":
                        parsed = self._parse_label_call(call, frame_var)
                        if parsed:
                            labels.append(parsed)
                    elif widget in WIDGET_TYPE_MAP:
                        parsed = self._parse_widget_call(call, widget, frame_var)
                        if parsed:
                            fields.append(parsed)

    def _parse_label_call(self, call: ast.Call, default_frame: str) -> ParsedLabel | None:
        if len(call.args) < 4:
            return None

        frame_var = _get_name(call.args[0]) or default_frame
        row = _get_int_or_lineno(call.args[1], call)
        col = _get_int(call.args[2]) if len(call.args) > 2 else None
        col = col if col is not None else 0
        text = _get_str(call.args[3])
        if not text:
            return None
        tooltip = _get_kwarg_str(call, "tooltip")

        return ParsedLabel(frame_var=frame_var, row=row, col=col, text=text, tooltip=tooltip)

    def _parse_widget_call(self, call: ast.Call, widget: str, default_frame: str) -> ParsedField | None:
        if len(call.args) < 3:
            return None

        frame_var = _get_name(call.args[0]) or default_frame
        row = _get_int_or_lineno(call.args[1], call)
        col = _get_int(call.args[2]) if len(call.args) > 2 else None
        col = col if col is not None else 0

        field = ParsedField(
            frame_var=frame_var,
            row=row,
            col=col,
            key="",
            widget_type=WIDGET_TYPE_MAP.get(widget, "entry"),
        )

        if widget == "entry":
            if len(call.args) >= 5:
                field.key = _get_str(call.args[4]) or ""
            field.tooltip = _get_kwarg_str(call, "tooltip")
            field.required = _get_kwarg_bool(call, "required") or False

        elif widget == "switch":
            if len(call.args) >= 5:
                field.key = _get_str(call.args[4]) or ""
            elif len(call.args) >= 4:
                key = _get_kwarg_str(call, "var_name")
                field.key = key or ""

        elif widget == "options":
            if len(call.args) >= 6:
                field.key = _get_str(call.args[5]) or ""
                field.enum_ref = _extract_enum_ref(call.args[3])

        elif widget == "options_adv":
            if len(call.args) >= 6:
                field.key = _get_str(call.args[5]) or ""
                field.enum_ref = _extract_enum_ref(call.args[3])
            field.adv_command = _extract_adv_command_name(call)

        elif widget == "options_kv":
            if len(call.args) >= 6:
                field.key = _get_str(call.args[5]) or ""
                field.kv_options = _extract_kv_options(call.args[3], self._local_lists)
                if field.kv_options is None:
                    field.dtype_subset = _extract_dtype_subset(call.args[3])
            field.widget_type = "select-kv"

        elif widget == "path_entry":
            if len(call.args) >= 5:
                field.key = _get_str(call.args[4]) or ""
            mode = _get_kwarg_str(call, "mode")
            field.path_mode = mode or "file"
            field.widget_type = "dir" if mode == "dir" else "file"

        elif widget == "time_entry":
            if len(call.args) >= 5:
                field.key = _get_str(call.args[4]) or ""
            if len(call.args) >= 6:
                field.unit_var = _get_str(call.args[5]) or ""

        elif widget == "layer_filter_entry":
            if len(call.args) >= 5:
                field.preset_var = _get_str(call.args[4]) or ""
            field.key = _get_kwarg_str(call, "entry_var_name") or field.preset_var or "layer_filter"
            field.regex_var = _get_kwarg_str(call, "regex_var_name")

        if not field.key:
            logger.debug("Could not extract key from %s call at line %d", widget, call.lineno)
            return None

        return field


def _label_from_key(key: str) -> str:
    name = key.split(".")[-1]
    words = name.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words)


def parse_ui_file(filepath: str | Path) -> ParsedTab | None:
    parser = CtkTabParser()
    return parser.parse_file(filepath)
