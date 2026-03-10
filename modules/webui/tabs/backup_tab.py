"""Backup settings tab for the Gradio WebUI."""

import gradio as gr

from modules.util.enum.TimeUnit import TimeUnit


def create_backup_tab():
    """Build the 'backup' tab and return a dict of components keyed by config attr."""
    components = {}

    with gr.Row():
        # ── left column ────────────────────────────────────────────
        with gr.Column():
            with gr.Row():
                components["backup_after"] = gr.Number(
                    label="Backup After",
                    value=30,
                    precision=0,
                    info="Create a full backup after a set time",
                    interactive=True,
                )
                components["backup_after_unit"] = gr.Dropdown(
                    label="Unit",
                    choices=[str(x) for x in list(TimeUnit)],
                    value=str(TimeUnit.MINUTE),
                    interactive=True,
                )

            components["rolling_backup"] = gr.Checkbox(
                label="Rolling Backup",
                value=False,
                info="Only keeps one backup at a time",
                interactive=True,
            )
            components["rolling_backup_count"] = gr.Number(
                label="Rolling Backup Count",
                value=3,
                precision=0,
                info="The number of backups to keep if rolling backup is enabled",
                interactive=True,
            )
            components["backup_before_save"] = gr.Checkbox(
                label="Backup Before Save",
                value=True,
                info="Creates a backup before saving the final output model",
                interactive=True,
            )

        # ── right column ───────────────────────────────────────────
        with gr.Column():
            with gr.Row():
                components["save_after"] = gr.Number(
                    label="Save After",
                    value=0,
                    precision=0,
                    info="Saves the model at the set interval. Uses the same destination as the final model",
                    interactive=True,
                )
                components["save_after_unit"] = gr.Dropdown(
                    label="Unit",
                    choices=[str(x) for x in list(TimeUnit)],
                    value=str(TimeUnit.NEVER),
                    interactive=True,
                )

            components["save_filename_prefix"] = gr.Textbox(
                label="Save Filename Prefix",
                value="",
                info="A prefix appended to the save filename",
                interactive=True,
            )

    return components
