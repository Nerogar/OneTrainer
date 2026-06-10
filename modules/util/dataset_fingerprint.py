"""SHA-256 fingerprint of the configured dataset, warn-only on resume mismatch."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

from modules.util.config.ConceptConfig import ConceptConfig


def _identifier_tuple(c) -> tuple:
    def g(name, default):
        if hasattr(c, name):
            return getattr(c, name)
        if isinstance(c, dict):
            return c.get(name, default)
        return default

    raw_type = g('type', '')
    type_str = getattr(raw_type, 'value', raw_type)
    return (
        str(g('name', '') or ''),
        str(g('path', '') or ''),
        int(g('seed', 0) or 0),
        str(type_str or ''),
        bool(g('include_subdirectories', False)),
        bool(g('enabled', True)),
    )


def compute_concept_fingerprint(
        concepts: Iterable[ConceptConfig] | Iterable[dict] | None,
        concept_file_name: str | None = None,
) -> tuple[str, int]:
    items: list = []
    if concepts:
        items = list(concepts)
    elif concept_file_name and os.path.exists(concept_file_name):
        # Mirrors TrainConfig.to_pack_dict: under the GUI, concepts live in a file.
        try:
            with open(concept_file_name, 'r') as f:
                items = json.load(f) or []
        except (OSError, ValueError):
            items = []

    payload = [_identifier_tuple(c) for c in items]
    payload.sort(key=lambda t: t[1])
    blob = json.dumps(payload, separators=(',', ':'), sort_keys=False).encode()
    return hashlib.sha256(blob).hexdigest(), len(payload)
