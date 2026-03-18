"""General settings tab for the Gradio WebUI."""

import gradio as gr

from modules.util.enum.TimeUnit import TimeUnit


def create_general_tab():
    """Build the 'general' tab and return a dict of components keyed by config attr."""
    components = {}

    with gr.Row():
        # ── left column ────────────────────────────────────────────
        with gr.Column():
            components["workspace_dir"] = gr.Textbox(
                label="Workspace Directory",
                value="workspace/run",
                info="The directory used for the workspace",
                interactive=True,
            )
            components["cache_dir"] = gr.Textbox(
                label="Cache Directory",
                value="workspace-cache/run",
                info="The directory used for caching",
                interactive=True,
            )
            components["continue_last_backup"] = gr.Checkbox(
                label="Continue Last Backup",
                value=True,
                info="Automatically continue training from the last backup saved in the workspace directory",
                interactive=True,
            )
            components["only_cache"] = gr.Checkbox(
                label="Only Cache",
                value=False,
                info="Only populate the cache with latent images, then stop",
                interactive=True,
            )

        # ── right column ───────────────────────────────────────────
        with gr.Column():
            components["debug_mode"] = gr.Dropdown(
                label="Debug Mode",
                choices=["NONE", "DETERMINISTIC", "NORMALIZED"],
                value="NONE",
                info="Saves debug information to the workspace directory during training",
                interactive=True,
            )
            components["debug_dir"] = gr.Textbox(
                label="Debug Directory",
                value="debug",
                info="The directory used for debug output",
                interactive=True,
            )
            components["tensorboard"] = gr.Checkbox(
                label="Tensorboard",
                value=True,
                info="Enable tensorboard logging during training",
                interactive=True,
            )
            components["tensorboard_expose"] = gr.Checkbox(
                label="Tensorboard Expose",
                value=False,
                info="Expose tensorboard to the network",
                interactive=True,
            )
            components["tensorboard_always_on"] = gr.Checkbox(
                label="Tensorboard Always On",
                value=False,
                info="Start Tensorboard as soon as OneTrainer loads and leave it running until exit",
                interactive=True,
            )
            components["tensorboard_port"] = gr.Number(
                label="Tensorboard Port",
                value=6006,
                precision=0,
                info="Port Tensorboard will receive connections from",
                interactive=True,
            )

            with gr.Row():
                components["validate_after"] = gr.Number(
                    label="Validate After",
                    value=0,
                    precision=0,
                    info="Do a validation run after a set time",
                    interactive=True,
                )
                components["validate_after_unit"] = gr.Dropdown(
                    label="Unit",
                    choices=[str(x) for x in list(TimeUnit)],
                    value=str(TimeUnit.NEVER),
                    interactive=True,
                )

    return components
