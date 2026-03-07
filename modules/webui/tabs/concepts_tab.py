"""Concepts tab for the Gradio WebUI — Fixed-slot architecture.

Uses a MAX_CONCEPTS number of pre-created concept card slots, showing/hiding
them based on how many concepts are active. This avoids @gr.render which
causes KeyError issues due to dynamic component creation/destruction.

Each slot provides image preview, prompt display, and full settings matching
the desktop UI ConceptWindow.
"""

import os
import gradio as gr

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.enum.ConceptType import ConceptType
from modules.webui.config_io import (
    get_dataset_images,
    get_image_prompt,
    load_image_thumbnail,
    load_concepts_from_file,
    save_concepts_to_file,
    ensure_concept_file_exists,
)

MAX_CONCEPTS = 10


def _new_concept_dict() -> dict:
    """Create a default ConceptConfig and return it as a dict."""
    return ConceptConfig.default_values().to_dict()


def _scan_preview(concept_dict: dict, image_index: int = 0):
    """Scan dataset path for images and return preview info.

    Returns (image_path_or_None, filename_str, prompt_str, clamped_index, total_count).
    """
    path = concept_dict.get("path", "")
    include_subdirs = concept_dict.get("include_subdirectories", False)

    images = get_dataset_images(path, include_subdirs)
    total = len(images)

    if total == 0:
        return None, "No images found", "", 0, 0

    idx = max(0, min(image_index, total - 1))
    img_path = images[idx]
    filename = os.path.basename(img_path)

    text_cfg = concept_dict.get("text", {})
    prompt_source = text_cfg.get("prompt_source", "sample")
    prompt_path = text_cfg.get("prompt_path", "")
    prompt = get_image_prompt(img_path, prompt_source, prompt_path)

    thumb = load_image_thumbnail(img_path)
    return thumb, filename, prompt, idx, total


def create_concepts_tab():
    """Build the 'concepts' tab with fixed-slot architecture.

    Returns dict of components for wiring in app.py.
    """
    components_dict = {}

    # ── State ───────────────────────────────────────────────────────
    concept_list = gr.State([])  # list[dict] of concept data
    concept_file = gr.State("training_concepts/concepts.json")
    # Per-slot image navigation index
    preview_indices = gr.State([0] * MAX_CONCEPTS)

    components_dict["_concept_list_state"] = concept_list
    components_dict["_concept_file_state"] = concept_file

    # ── Top controls ────────────────────────────────────────────────
    with gr.Row():
        add_btn = gr.Button("➕ Add Concept", variant="primary", scale=1)
        save_concepts_btn = gr.Button("💾 Save Concepts", variant="secondary", scale=0, min_width=140)
        status_msg = gr.Textbox(label="Status", value="", interactive=False, scale=2)

    components_dict["_concept_add_btn"] = add_btn
    components_dict["_concept_status"] = status_msg

    # ── Pre-create concept slots ────────────────────────────────────
    slot_groups = []       # gr.Group for each slot
    slot_components = []   # dict of all components per slot

    for slot_idx in range(MAX_CONCEPTS):
        visible = False  # All hidden initially

        with gr.Group(visible=visible) as group:
            slot = {}

            # ── Header row: enabled, name, path, remove ────────────
            with gr.Row():
                slot["enabled"] = gr.Checkbox(
                    label="Enabled", value=True, scale=0, min_width=80, interactive=True,
                )
                slot["name"] = gr.Textbox(
                    label="Name", value="", scale=2, interactive=True,
                )
                slot["path"] = gr.Textbox(
                    label="Path", value="", scale=3, interactive=True,
                )
                slot["remove_btn"] = gr.Button("🗑 Remove", variant="stop", scale=0, min_width=100)

            # ── Preview row ─────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    slot["preview_image"] = gr.Image(
                        label="Preview", value=None, height=200, width=200,
                        interactive=False,
                    )
                    with gr.Row():
                        slot["prev_btn"] = gr.Button("◀", scale=0, min_width=50)
                        slot["scan_btn"] = gr.Button("🔍 Scan", variant="secondary", scale=1)
                        slot["next_btn"] = gr.Button("▶", scale=0, min_width=50)

                with gr.Column(scale=2):
                    slot["preview_filename"] = gr.Textbox(
                        label="Filename", value="No images found", interactive=False, lines=1,
                    )
                    slot["preview_prompt"] = gr.Textbox(
                        label="Prompt", value="", interactive=False, lines=3,
                    )
                    slot["preview_info"] = gr.Textbox(
                        label="Dataset", value="", interactive=False, lines=1,
                    )

            # ── Settings accordion ──────────────────────────────────
            with gr.Accordion(f"Concept {slot_idx + 1} — Settings", open=False):
                with gr.Tab("General"):
                    with gr.Row():
                        slot["type"] = gr.Dropdown(
                            label="Concept Type",
                            choices=[str(x) for x in list(ConceptType)],
                            value=str(ConceptType.STANDARD),
                            interactive=True,
                        )
                        slot["balancing_strategy"] = gr.Dropdown(
                            label="Balancing Strategy",
                            choices=[str(x) for x in list(BalancingStrategy)],
                            value=str(BalancingStrategy.REPEATS),
                            interactive=True,
                        )
                        slot["balancing"] = gr.Number(
                            label="Balancing", value=1.0, interactive=True,
                        )
                    with gr.Row():
                        slot["prompt_source"] = gr.Dropdown(
                            label="Prompt Source",
                            choices=[
                                ("From text file per sample", "sample"),
                                ("From single text file", "concept"),
                                ("From image file name", "filename"),
                            ],
                            value="sample",
                            interactive=True,
                        )
                        slot["prompt_path"] = gr.Textbox(
                            label="Prompt Path", value="", interactive=True,
                        )
                    with gr.Row():
                        slot["include_subdirectories"] = gr.Checkbox(
                            label="Include Subdirectories", value=False, interactive=True,
                        )
                        slot["image_variations"] = gr.Number(
                            label="Image Variations", value=1, precision=0, interactive=True,
                        )
                        slot["text_variations"] = gr.Number(
                            label="Text Variations", value=1, precision=0, interactive=True,
                        )
                        slot["loss_weight"] = gr.Number(
                            label="Loss Weight", value=1.0, interactive=True,
                        )

                with gr.Tab("Image Augmentation"):
                    with gr.Row():
                        slot["img_crop_jitter"] = gr.Checkbox(label="Crop Jitter", value=True, interactive=True)
                        slot["img_random_flip"] = gr.Checkbox(label="Random Flip", value=False, interactive=True)
                        slot["img_fixed_flip"] = gr.Checkbox(label="Fixed Flip", value=False, interactive=True)
                    with gr.Row():
                        slot["img_random_rotate"] = gr.Checkbox(label="Random Rotate", value=False, interactive=True)
                        slot["img_fixed_rotate"] = gr.Checkbox(label="Fixed Rotate", value=False, interactive=True)
                        slot["img_rotate_max"] = gr.Number(label="Max Angle", value=0.0, interactive=True)
                    with gr.Row():
                        slot["img_random_brightness"] = gr.Checkbox(label="Random Brightness", value=False, interactive=True)
                        slot["img_fixed_brightness"] = gr.Checkbox(label="Fixed Brightness", value=False, interactive=True)
                        slot["img_brightness_max"] = gr.Number(label="Max Brightness", value=0.0, interactive=True)
                    with gr.Row():
                        slot["img_random_contrast"] = gr.Checkbox(label="Random Contrast", value=False, interactive=True)
                        slot["img_fixed_contrast"] = gr.Checkbox(label="Fixed Contrast", value=False, interactive=True)
                        slot["img_contrast_max"] = gr.Number(label="Max Contrast", value=0.0, interactive=True)
                    with gr.Row():
                        slot["img_random_saturation"] = gr.Checkbox(label="Random Saturation", value=False, interactive=True)
                        slot["img_fixed_saturation"] = gr.Checkbox(label="Fixed Saturation", value=False, interactive=True)
                        slot["img_saturation_max"] = gr.Number(label="Max Saturation", value=0.0, interactive=True)
                    with gr.Row():
                        slot["img_random_hue"] = gr.Checkbox(label="Random Hue", value=False, interactive=True)
                        slot["img_fixed_hue"] = gr.Checkbox(label="Fixed Hue", value=False, interactive=True)
                        slot["img_hue_max"] = gr.Number(label="Max Hue", value=0.0, interactive=True)
                    with gr.Row():
                        slot["img_resolution_override"] = gr.Checkbox(label="Resolution Override", value=False, interactive=True)
                        slot["img_resolution_value"] = gr.Textbox(label="Resolution", value="512", interactive=True)
                    with gr.Row():
                        slot["img_circular_mask"] = gr.Checkbox(label="Circular Mask Shrink", value=False, interactive=True)
                        slot["img_mask_rotate_crop"] = gr.Checkbox(label="Mask Rotate Crop", value=False, interactive=True)

                with gr.Tab("Text Augmentation"):
                    with gr.Row():
                        slot["txt_tag_shuffling"] = gr.Checkbox(label="Tag Shuffling", value=False, interactive=True)
                        slot["txt_tag_delimiter"] = gr.Textbox(label="Tag Delimiter", value=",", interactive=True)
                        slot["txt_keep_tags"] = gr.Number(label="Keep Tags Count", value=1, precision=0, interactive=True)
                    with gr.Row():
                        slot["txt_dropout_enable"] = gr.Checkbox(label="Tag Dropout", value=False, interactive=True)
                        slot["txt_dropout_mode"] = gr.Dropdown(
                            label="Dropout Mode",
                            choices=["FULL", "RANDOM", "RANDOM WEIGHTED"],
                            value="FULL", interactive=True,
                        )
                        slot["txt_dropout_prob"] = gr.Number(label="Dropout Probability", value=0.0, interactive=True)
                    with gr.Row():
                        slot["txt_caps_enable"] = gr.Checkbox(label="Caps Randomize", value=False, interactive=True)
                        slot["txt_caps_prob"] = gr.Number(label="Caps Probability", value=0.0, interactive=True)
                        slot["txt_caps_lowercase"] = gr.Checkbox(label="Force Lowercase", value=False, interactive=True)

            slot_groups.append(group)
            slot_components.append(slot)

    components_dict["_slot_groups"] = slot_groups
    components_dict["_slot_components"] = slot_components

    # ════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ════════════════════════════════════════════════════════════════

    # Field mapping: slot key → concept dict path
    FLAT_FIELDS = {
        "name": "name",
        "path": "path",
        "enabled": "enabled",
        "type": "type",
        "balancing_strategy": "balancing_strategy",
        "balancing": "balancing",
        "loss_weight": "loss_weight",
        "include_subdirectories": "include_subdirectories",
        "image_variations": "image_variations",
        "text_variations": "text_variations",
    }

    IMG_FIELDS = {
        "img_crop_jitter": "enable_crop_jitter",
        "img_random_flip": "enable_random_flip",
        "img_fixed_flip": "enable_fixed_flip",
        "img_random_rotate": "enable_random_rotate",
        "img_fixed_rotate": "enable_fixed_rotate",
        "img_rotate_max": "random_rotate_max_angle",
        "img_random_brightness": "enable_random_brightness",
        "img_fixed_brightness": "enable_fixed_brightness",
        "img_brightness_max": "random_brightness_max_strength",
        "img_random_contrast": "enable_random_contrast",
        "img_fixed_contrast": "enable_fixed_contrast",
        "img_contrast_max": "random_contrast_max_strength",
        "img_random_saturation": "enable_random_saturation",
        "img_fixed_saturation": "enable_fixed_saturation",
        "img_saturation_max": "random_saturation_max_strength",
        "img_random_hue": "enable_random_hue",
        "img_fixed_hue": "enable_fixed_hue",
        "img_hue_max": "random_hue_max_strength",
        "img_resolution_override": "enable_resolution_override",
        "img_resolution_value": "resolution_override",
        "img_circular_mask": "enable_random_circular_mask_shrink",
        "img_mask_rotate_crop": "enable_random_mask_rotate_crop",
    }

    TXT_FIELDS = {
        "txt_tag_shuffling": "enable_tag_shuffling",
        "txt_tag_delimiter": "tag_delimiter",
        "txt_keep_tags": "keep_tags_count",
        "txt_dropout_enable": "tag_dropout_enable",
        "txt_dropout_mode": "tag_dropout_mode",
        "txt_dropout_prob": "tag_dropout_probability",
        "txt_caps_enable": "caps_randomize_enable",
        "txt_caps_prob": "caps_randomize_probability",
        "txt_caps_lowercase": "caps_randomize_lowercase",
    }

    PROMPT_FIELDS = {
        "prompt_source": "prompt_source",
        "prompt_path": "prompt_path",
    }

    # ── Visibility update outputs ───────────────────────────────────
    def _visibility_updates(concepts):
        """Return gr.update(visible=...) for each slot group."""
        return [gr.update(visible=(i < len(concepts))) for i in range(MAX_CONCEPTS)]

    # ── Populate slot fields from concept dict ─────────────────────
    def _populate_slot_outputs(concepts):
        """Return flattened list of gr.update() for all slot fields."""
        updates = []
        for i in range(MAX_CONCEPTS):
            if i < len(concepts):
                c = concepts[i]
                img = c.get("image", {})
                txt = c.get("text", {})
                updates.extend([
                    # Flat fields
                    gr.update(value=c.get("name", "")),
                    gr.update(value=c.get("path", "")),
                    gr.update(value=c.get("enabled", True)),
                    gr.update(value=c.get("type", str(ConceptType.STANDARD))),
                    gr.update(value=c.get("balancing_strategy", str(BalancingStrategy.REPEATS))),
                    gr.update(value=c.get("balancing", 1.0)),
                    gr.update(value=c.get("loss_weight", 1.0)),
                    gr.update(value=c.get("include_subdirectories", False)),
                    gr.update(value=c.get("image_variations", 1)),
                    gr.update(value=c.get("text_variations", 1)),
                    # Prompt fields (in text)
                    gr.update(value=txt.get("prompt_source", "sample")),
                    gr.update(value=txt.get("prompt_path", "")),
                    # Image aug fields
                    gr.update(value=img.get("enable_crop_jitter", True)),
                    gr.update(value=img.get("enable_random_flip", False)),
                    gr.update(value=img.get("enable_fixed_flip", False)),
                    gr.update(value=img.get("enable_random_rotate", False)),
                    gr.update(value=img.get("enable_fixed_rotate", False)),
                    gr.update(value=img.get("random_rotate_max_angle", 0.0)),
                    gr.update(value=img.get("enable_random_brightness", False)),
                    gr.update(value=img.get("enable_fixed_brightness", False)),
                    gr.update(value=img.get("random_brightness_max_strength", 0.0)),
                    gr.update(value=img.get("enable_random_contrast", False)),
                    gr.update(value=img.get("enable_fixed_contrast", False)),
                    gr.update(value=img.get("random_contrast_max_strength", 0.0)),
                    gr.update(value=img.get("enable_random_saturation", False)),
                    gr.update(value=img.get("enable_fixed_saturation", False)),
                    gr.update(value=img.get("random_saturation_max_strength", 0.0)),
                    gr.update(value=img.get("enable_random_hue", False)),
                    gr.update(value=img.get("enable_fixed_hue", False)),
                    gr.update(value=img.get("random_hue_max_strength", 0.0)),
                    gr.update(value=img.get("enable_resolution_override", False)),
                    gr.update(value=img.get("resolution_override", "512")),
                    gr.update(value=img.get("enable_random_circular_mask_shrink", False)),
                    gr.update(value=img.get("enable_random_mask_rotate_crop", False)),
                    # Text aug fields
                    gr.update(value=txt.get("enable_tag_shuffling", False)),
                    gr.update(value=txt.get("tag_delimiter", ",")),
                    gr.update(value=txt.get("keep_tags_count", 1)),
                    gr.update(value=txt.get("tag_dropout_enable", False)),
                    gr.update(value=txt.get("tag_dropout_mode", "FULL")),
                    gr.update(value=txt.get("tag_dropout_probability", 0.0)),
                    gr.update(value=txt.get("caps_randomize_enable", False)),
                    gr.update(value=txt.get("caps_randomize_probability", 0.0)),
                    gr.update(value=txt.get("caps_randomize_lowercase", False)),
                ])
            else:
                # Hidden slot — send no-op updates
                updates.extend([gr.update()] * 43)  # 10 flat + 2 prompt + 22 img + 9 txt = 43
        return updates

    # Build output lists for visibility + populate
    def _all_slot_field_components():
        """Return flattened list of all slot field components in order."""
        result = []
        for slot in slot_components:
            # Must match the order in _populate_slot_outputs
            result.extend([
                slot["name"], slot["path"], slot["enabled"],
                slot["type"], slot["balancing_strategy"], slot["balancing"],
                slot["loss_weight"], slot["include_subdirectories"],
                slot["image_variations"], slot["text_variations"],
                slot["prompt_source"], slot["prompt_path"],
                # Image aug
                slot["img_crop_jitter"], slot["img_random_flip"], slot["img_fixed_flip"],
                slot["img_random_rotate"], slot["img_fixed_rotate"], slot["img_rotate_max"],
                slot["img_random_brightness"], slot["img_fixed_brightness"], slot["img_brightness_max"],
                slot["img_random_contrast"], slot["img_fixed_contrast"], slot["img_contrast_max"],
                slot["img_random_saturation"], slot["img_fixed_saturation"], slot["img_saturation_max"],
                slot["img_random_hue"], slot["img_fixed_hue"], slot["img_hue_max"],
                slot["img_resolution_override"], slot["img_resolution_value"],
                slot["img_circular_mask"], slot["img_mask_rotate_crop"],
                # Text aug
                slot["txt_tag_shuffling"], slot["txt_tag_delimiter"], slot["txt_keep_tags"],
                slot["txt_dropout_enable"], slot["txt_dropout_mode"], slot["txt_dropout_prob"],
                slot["txt_caps_enable"], slot["txt_caps_prob"], slot["txt_caps_lowercase"],
            ])
        return result

    all_field_outputs = _all_slot_field_components()
    all_vis_outputs = list(slot_groups)

    def _full_refresh(concepts):
        """Return visibility + field updates for all slots."""
        vis = _visibility_updates(concepts)
        fields = _populate_slot_outputs(concepts)
        return [concepts] + vis + fields

    full_outputs = [concept_list] + all_vis_outputs + all_field_outputs

    # ── ADD CONCEPT ─────────────────────────────────────────────────
    def _add_concept(concepts):
        if len(concepts) >= MAX_CONCEPTS:
            return [concepts] + _visibility_updates(concepts) + _populate_slot_outputs(concepts)
        new_concepts = concepts + [_new_concept_dict()]
        return _full_refresh(new_concepts)

    add_btn.click(fn=_add_concept, inputs=[concept_list], outputs=full_outputs)

    # ── REMOVE CONCEPT (per slot) ───────────────────────────────────
    for idx in range(MAX_CONCEPTS):
        def _make_remove(i):
            def _remove(concepts):
                new_concepts = list(concepts)
                if i < len(new_concepts):
                    new_concepts.pop(i)
                return _full_refresh(new_concepts)
            return _remove

        slot_components[idx]["remove_btn"].click(
            fn=_make_remove(idx),
            inputs=[concept_list],
            outputs=full_outputs,
        )

    # ── FIELD CHANGE → update concept dict in state ─────────────────
    for idx in range(MAX_CONCEPTS):
        slot = slot_components[idx]

        # Flat fields
        for slot_key, dict_key in FLAT_FIELDS.items():
            def _make_update_flat(i, dk):
                def _update(concepts, value):
                    if i < len(concepts):
                        concepts = [dict(c) for c in concepts]
                        concepts[i][dk] = value
                    return concepts
                return _update

            slot[slot_key].change(
                fn=_make_update_flat(idx, dict_key),
                inputs=[concept_list, slot[slot_key]],
                outputs=[concept_list],
            )

        # Prompt fields (nested under "text")
        for slot_key, dict_key in PROMPT_FIELDS.items():
            def _make_update_txt_prompt(i, dk):
                def _update(concepts, value):
                    if i < len(concepts):
                        concepts = [dict(c) for c in concepts]
                        text = dict(concepts[i].get("text", {}))
                        text[dk] = value
                        concepts[i]["text"] = text
                    return concepts
                return _update

            slot[slot_key].change(
                fn=_make_update_txt_prompt(idx, dict_key),
                inputs=[concept_list, slot[slot_key]],
                outputs=[concept_list],
            )

        # Image augmentation fields
        for slot_key, dict_key in IMG_FIELDS.items():
            def _make_update_img(i, dk):
                def _update(concepts, value):
                    if i < len(concepts):
                        concepts = [dict(c) for c in concepts]
                        img = dict(concepts[i].get("image", {}))
                        img[dk] = value
                        concepts[i]["image"] = img
                    return concepts
                return _update

            slot[slot_key].change(
                fn=_make_update_img(idx, dict_key),
                inputs=[concept_list, slot[slot_key]],
                outputs=[concept_list],
            )

        # Text augmentation fields
        for slot_key, dict_key in TXT_FIELDS.items():
            def _make_update_txt(i, dk):
                def _update(concepts, value):
                    if i < len(concepts):
                        concepts = [dict(c) for c in concepts]
                        txt = dict(concepts[i].get("text", {}))
                        txt[dk] = value
                        concepts[i]["text"] = txt
                    return concepts
                return _update

            slot[slot_key].change(
                fn=_make_update_txt(idx, dict_key),
                inputs=[concept_list, slot[slot_key]],
                outputs=[concept_list],
            )

    # ── SCAN / PREVIEW (per slot) ───────────────────────────────────
    for idx in range(MAX_CONCEPTS):
        slot = slot_components[idx]

        def _make_scan(i):
            def _scan(concepts, indices):
                if i >= len(concepts):
                    return None, "No concept", "", "N/A"
                img_idx = indices[i] if i < len(indices) else 0
                thumb, fname, prompt, clamped, total = _scan_preview(concepts[i], img_idx)
                info = f"Image {clamped + 1} of {total}" if total > 0 else "No images found"
                return thumb, fname, prompt, info
            return _scan

        slot["scan_btn"].click(
            fn=_make_scan(idx),
            inputs=[concept_list, preview_indices],
            outputs=[slot["preview_image"], slot["preview_filename"],
                     slot["preview_prompt"], slot["preview_info"]],
        )

        # Auto-scan when path changes
        def _make_path_scan(i):
            def _path_scan(concepts, indices):
                if i >= len(concepts):
                    return None, "No concept", "", "N/A", indices
                # Reset index to 0 on path change
                new_indices = list(indices)
                new_indices[i] = 0
                thumb, fname, prompt, clamped, total = _scan_preview(concepts[i], 0)
                info = f"Image {clamped + 1} of {total}" if total > 0 else "No images found"
                return thumb, fname, prompt, info, new_indices
            return _path_scan

        slot["path"].change(
            fn=_make_path_scan(idx),
            inputs=[concept_list, preview_indices],
            outputs=[slot["preview_image"], slot["preview_filename"],
                     slot["preview_prompt"], slot["preview_info"], preview_indices],
        )

        # Prev/Next navigation
        def _make_nav(i, delta):
            def _nav(concepts, indices):
                if i >= len(concepts):
                    return None, "", "", "N/A", indices
                new_indices = list(indices)
                new_indices[i] = max(0, new_indices[i] + delta)
                thumb, fname, prompt, clamped, total = _scan_preview(concepts[i], new_indices[i])
                new_indices[i] = clamped
                info = f"Image {clamped + 1} of {total}" if total > 0 else "No images found"
                return thumb, fname, prompt, info, new_indices
            return _nav

        slot["prev_btn"].click(
            fn=_make_nav(idx, -1),
            inputs=[concept_list, preview_indices],
            outputs=[slot["preview_image"], slot["preview_filename"],
                     slot["preview_prompt"], slot["preview_info"], preview_indices],
        )
        slot["next_btn"].click(
            fn=_make_nav(idx, +1),
            inputs=[concept_list, preview_indices],
            outputs=[slot["preview_image"], slot["preview_filename"],
                     slot["preview_prompt"], slot["preview_info"], preview_indices],
        )

    # ── SAVE CONCEPTS ───────────────────────────────────────────────
    def _save_concepts_click(concepts, cfile):
        try:
            save_concepts_to_file(concepts, cfile)
            return f"✅ Saved {len(concepts)} concept(s) to {cfile}"
        except Exception as e:
            return f"❌ Error: {e}"

    save_concepts_btn.click(
        fn=_save_concepts_click,
        inputs=[concept_list, concept_file],
        outputs=[status_msg],
    )

    # Store refresh function for external use (app.py load)
    components_dict["_full_refresh_fn"] = _full_refresh
    components_dict["_full_outputs"] = full_outputs

    return components_dict
