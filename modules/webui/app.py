"""
Main Gradio application factory for the OneTrainer WebUI.

Creates a gr.Blocks app with:
  - Top bar: config preset, save/load, model type, training method, start/stop
  - Tabs: general, model, data, concepts, training, sampling, backup,
          tools, additional embeddings, lora (conditional), cloud
  - Bottom bar: progress, status, ETA, action buttons, theme toggle
  - gr.Timer for polling training state
"""

import json
import base64
import webbrowser
from pathlib import Path

# Embed logo as base64 data URI so it works everywhere without file-serving issues
_logo_file = Path("resources/icons/icon.png").resolve()
if _logo_file.exists():
    _logo_b64 = base64.b64encode(_logo_file.read_bytes()).decode()
    LOGO_DATA_URI = f"data:image/png;base64,{_logo_b64}"
else:
    LOGO_DATA_URI = ""

# Keep directory export for allowed_paths (not strictly needed now but harmless)
LOGO_DIR = str(_logo_file.parent)

import gradio as gr

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.webui import webui_bridge
from modules.webui.config_io import (
    ensure_concept_file_exists,
    ensure_sample_file_exists,
    list_presets,
    load_concepts_from_file,
    load_config_from_file,
    load_samples_from_file,
    preset_path,
    save_concepts_to_file,
    save_config_to_file,
    save_samples_to_file,
)
from modules.webui.tabs.backup_tab import create_backup_tab
from modules.webui.tabs.cloud_tab import create_cloud_tab
from modules.webui.tabs.concepts_tab import create_concepts_tab
from modules.webui.tabs.data_tab import create_data_tab
from modules.webui.tabs.embeddings_tab import create_embeddings_tab
from modules.webui.tabs.general_tab import create_general_tab
from modules.webui.tabs.lora_tab import create_lora_tab
from modules.webui.tabs.model_tab import create_model_tab, update_model_tab_visibility
from modules.webui.tabs.sampling_tab import create_sampling_tab
from modules.webui.tabs.tools_tab import create_tools_tab
from modules.webui.tabs.training_tab import create_training_tab
from modules.webui.webui_state import get_state


# ── Helpers ─────────────────────────────────────────────────────────────────

def _training_methods_for(model_type_str: str) -> list[str]:
    """Return valid TrainingMethod names for a given ModelType string."""
    try:
        mt = ModelType[model_type_str]
    except KeyError:
        mt = ModelType.STABLE_DIFFUSION_15

    all_methods = list(TrainingMethod)
    valid = []
    for m in all_methods:
        valid.append(str(m))
    return valid


# ── App factory ─────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks application."""
    state = get_state()
    preset_names = list_presets()

    # Ensure the concept and sample files exist on disk before building the UI
    concept_file = state.train_config.concept_file_name
    ensure_concept_file_exists(concept_file)
    initial_concepts = load_concepts_from_file(concept_file)

    sample_file = state.train_config.sample_definition_file_name
    ensure_sample_file_exists(sample_file)

    with gr.Blocks(
        title="OneTrainer WebUI",
    ) as app:

        # ════════════════════════════════════════════════════════════
        # TOP BAR
        # ════════════════════════════════════════════════════════════
        with gr.Row():
            config_preset = gr.Dropdown(
                label="Config Preset",
                choices=preset_names,
                value=preset_names[0] if preset_names else "< default >",
                scale=2,
                interactive=True,
            )
            save_btn = gr.Button("💾 Save", variant="secondary", scale=0, min_width=80)
            load_btn = gr.Button("📂 Load", variant="secondary", scale=0, min_width=80)
            wiki_btn = gr.Button("📖 Wiki", variant="secondary", scale=0, min_width=80)

            model_type = gr.Dropdown(
                label="Model Type",
                choices=[str(x) for x in list(ModelType)],
                value=str(ModelType.STABLE_DIFFUSION_15),
                scale=2,
                interactive=True,
            )
            training_method = gr.Dropdown(
                label="Training Method",
                choices=[str(x) for x in list(TrainingMethod)],
                value=str(TrainingMethod.FINE_TUNE),
                scale=2,
                interactive=True,
            )
            train_btn = gr.Button(
                "▶ start training",
                variant="primary",
                scale=1,
                min_width=160,
            )

            gr.HTML(
                f'<img src="{LOGO_DATA_URI}" height="40" width="40" '
                f'style="object-fit:contain; margin:auto 0;" alt="OneTrainer"/>'
            )

        # ════════════════════════════════════════════════════════════
        # TABS
        # ════════════════════════════════════════════════════════════
        with gr.Tabs():

            with gr.Tab("general"):
                general_comps = create_general_tab()

            with gr.Tab("model"):
                model_comps = create_model_tab()

            with gr.Tab("data"):
                data_comps = create_data_tab()

            with gr.Tab("concepts"):
                concepts_comps = create_concepts_tab()
                # Set initial concept data from disk by populating all slots
                _refresh_fn = concepts_comps["_full_refresh_fn"]
                _refresh_outputs = concepts_comps["_full_outputs"]
                app.load(
                    fn=lambda: _refresh_fn(initial_concepts),
                    outputs=_refresh_outputs,
                )

            with gr.Tab("training"):
                training_comps = create_training_tab()

            with gr.Tab("sampling"):
                sampling_comps = create_sampling_tab()

            with gr.Tab("backup"):
                backup_comps = create_backup_tab()

            with gr.Tab("tools"):
                tools_comps = create_tools_tab()

            with gr.Tab("additional embeddings"):
                embeddings_comps = create_embeddings_tab()

            with gr.Tab("lora", visible=False) as lora_tab_ref:
                lora_comps = create_lora_tab()

            with gr.Tab("cloud"):
                cloud_comps = create_cloud_tab()

        # ════════════════════════════════════════════════════════════
        # BOTTOM BAR
        # ════════════════════════════════════════════════════════════
        with gr.Row():
            progress_bar = gr.Slider(
                label="Progress",
                minimum=0,
                maximum=1,
                value=0,
                interactive=False,
                scale=4,
            )
            status_text = gr.Textbox(
                label="Status",
                value="idle",
                interactive=False,
                scale=2,
            )
            eta_text = gr.Textbox(
                label="ETA",
                value="",
                interactive=False,
                scale=1,
            )

        with gr.Row():
            sample_now_btn = gr.Button("sample now", variant="secondary")
            backup_now_btn = gr.Button("backup now", variant="secondary")
            save_now_btn = gr.Button("save now", variant="secondary")
            save_default_btn = gr.Button("save default", variant="secondary")
            theme_toggle_btn = gr.Button("🌙 Dark", variant="secondary", scale=0, min_width=110)

        # ── Inject logo into Gradio's built-in footer bar ────────────
        gr.HTML(f'''
        <script>
        (function injectFooterLogo() {{
            function tryInject() {{
                var footer = document.querySelector('footer');
                if (!footer) {{ setTimeout(tryInject, 300); return; }}
                if (footer.querySelector('.ot-footer-logo')) return;
                var container = document.createElement('span');
                container.className = 'ot-footer-logo';
                container.style.cssText = 'display:inline-flex; align-items:center; gap:4px;';
                container.innerHTML = '<img src="{LOGO_DATA_URI}" style="height:1em; width:auto; vertical-align:middle; opacity:0.8;"/> <span>OneTrainer</span> ·';
                footer.insertBefore(container, footer.firstChild);
            }}
            tryInject();
        }})();
        </script>
        ''')

        # ════════════════════════════════════════════════════════════
        # THEME TOGGLE (JavaScript-based dark/light switching)
        # ════════════════════════════════════════════════════════════
        gr.HTML("""
<script>
// Apply saved theme immediately on page load
(function() {
    const saved = localStorage.getItem('onetrainer_theme');
    if (saved === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
})();

function toggleOneTrainerTheme() {
    const root = document.documentElement;
    const isDark = root.classList.contains('dark');
    if (isDark) {
        root.classList.remove('dark');
        localStorage.setItem('onetrainer_theme', 'light');
    } else {
        root.classList.add('dark');
        localStorage.setItem('onetrainer_theme', 'dark');
    }
}
</script>
""")

        gr.HTML("""
<script>
// Sync button label to saved theme after page renders
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        const saved = localStorage.getItem('onetrainer_theme');
        const buttons = document.querySelectorAll('button');
        buttons.forEach(function(btn) {
            if (btn.textContent.trim() === '🌙 Dark' || btn.textContent.trim() === '☀️ Light') {
                btn.textContent = saved === 'dark' ? '☀️ Light' : '🌙 Dark';
            }
        });
    }, 500);
});
</script>
""")

        # ════════════════════════════════════════════════════════════
        # TIMER — polls training state every second
        # ════════════════════════════════════════════════════════════
        timer = gr.Timer(value=1.0, active=True)

        def _poll_and_update():
            info = webui_bridge.poll_state(state)
            return (
                info["progress"],       # progress_bar
                info["status"],         # status_text
                info["eta"],            # eta_text
                gr.update(             # train_btn
                    value=f"▶ {info['btn_label']}" if info["btn_label"] == "start training"
                          else f"⏹ {info['btn_label']}",
                    variant=info["btn_variant"],
                    interactive=info["btn_interactive"],
                ),
            )

        timer.tick(
            fn=_poll_and_update,
            outputs=[progress_bar, status_text, eta_text, train_btn],
        )

        # ════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ════════════════════════════════════════════════════════════

        # ── Start / Stop toggle ─────────────────────────────────────
        def _toggle_training(samples):
            # Write current samples to disk so to_pack_dict() can read them
            try:
                save_samples_to_file(
                    samples, state.train_config.sample_definition_file_name
                )
            except Exception:
                pass
            if state.running:
                webui_bridge.stop_training(state)
            else:
                webui_bridge.start_training(state)

        train_btn.click(
            fn=_toggle_training,
            inputs=[sampling_comps["_sample_list_state"]],
            outputs=None,
        )

        # ── Model type change → update training method choices ──────
        def _on_model_type_change(mt_str):
            methods = _training_methods_for(mt_str)
            return gr.update(choices=methods, value=methods[0] if methods else "")

        model_type.change(
            fn=_on_model_type_change,
            inputs=[model_type],
            outputs=[training_method],
        )

        # ── Training method change → lora tab visibility ────────────
        def _on_method_change(method_str):
            lora_visible = method_str in [
                str(TrainingMethod.LORA),
            ]
            if hasattr(TrainingMethod, "LORA_FINE_TUNE"):
                lora_visible = lora_visible or method_str == str(TrainingMethod.LORA_FINE_TUNE)
            return gr.update(visible=lora_visible)

        training_method.change(
            fn=_on_method_change,
            inputs=[training_method],
            outputs=[lora_tab_ref],
        )

        # ── Save / Load / Wiki ──────────────────────────────────────
        def _save_preset(name, concepts, cfile, samples):
            # 1. Write concepts to their own JSON file first
            try:
                save_concepts_to_file(concepts, cfile)
            except Exception as e:
                return f"Error saving concepts: {e}"
            # 2. Write samples to their JSON file
            try:
                save_samples_to_file(
                    samples, state.train_config.sample_definition_file_name
                )
            except Exception as e:
                return f"Error saving samples: {e}"
            # 3. Now save the training config (to_pack_dict reads concept+sample files)
            p = preset_path(name)
            try:
                save_config_to_file(state.train_config, p)
            except Exception as e:
                return f"Error saving config: {e}"
            return f"Saved to {p} ({len(concepts)} concepts, {len(samples)} samples)"

        def _load_preset(name):
            p = preset_path(name)
            if p.exists():
                state.train_config = load_config_from_file(p)
                # Reload concepts from concept file
                cfile = state.train_config.concept_file_name
                ensure_concept_file_exists(cfile)
                loaded_concepts = load_concepts_from_file(cfile)
                return [f"Loaded {p}"] + _refresh_fn(loaded_concepts)
            # Not found — return no-ops
            no_change = [gr.update()] * (len(_refresh_outputs))
            return [f"Preset not found: {p}"] + no_change

        save_btn.click(
            fn=_save_preset,
            inputs=[
                config_preset,
                concepts_comps["_concept_list_state"],
                concepts_comps["_concept_file_state"],
                sampling_comps["_sample_list_state"],
            ],
            outputs=[status_text],
        )
        load_btn.click(
            fn=_load_preset,
            inputs=[config_preset],
            outputs=[status_text] + _refresh_outputs,
        )
        wiki_btn.click(
            fn=lambda: webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki"),
            inputs=None,
            outputs=None,
        )

        # ── Bottom bar: runtime commands ────────────────────────────
        sample_now_btn.click(fn=lambda: webui_bridge.sample_now(state))
        backup_now_btn.click(fn=lambda: webui_bridge.backup_now(state))
        save_now_btn.click(fn=lambda: webui_bridge.save_now(state))

        def _save_default(concepts, cfile, samples):
            # Save concepts and samples first, then config
            try:
                save_concepts_to_file(concepts, cfile)
            except Exception:
                pass
            try:
                save_samples_to_file(
                    samples, state.train_config.sample_definition_file_name
                )
            except Exception:
                pass
            save_config_to_file(state.train_config, "training_presets/default.json")
            return "Default config saved"

        save_default_btn.click(
            fn=_save_default,
            inputs=[
                concepts_comps["_concept_list_state"],
                concepts_comps["_concept_file_state"],
                sampling_comps["_sample_list_state"],
            ],
            outputs=[status_text],
        )

        # ── Theme toggle ────────────────────────────────────────────
        theme_toggle_btn.click(
            fn=None,
            inputs=[],
            outputs=[theme_toggle_btn],
            js="""() => {
                const root = document.documentElement;
                const isDark = root.classList.contains('dark');
                if (isDark) {
                    root.classList.remove('dark');
                    localStorage.setItem('onetrainer_theme', 'light');
                    return '🌙 Dark';
                } else {
                    root.classList.add('dark');
                    localStorage.setItem('onetrainer_theme', 'dark');
                    return '☀️ Light';
                }
            }"""
        )

    return app
