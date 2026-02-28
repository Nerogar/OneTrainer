"""Tools tab for the Gradio WebUI."""

import gradio as gr


def create_tools_tab():
    """Build the 'tools' tab and return a dict of components keyed by purpose."""
    components = {}

    with gr.Column():
        components["btn_open_tensorboard"] = gr.Button(
            "Open Tensorboard",
            variant="secondary",
        )
        components["btn_export_config"] = gr.Button(
            "Export Training Config",
            variant="secondary",
        )

        with gr.Accordion("Profiling", open=False):
            gr.Markdown(
                "Profiling tools are only available when running the "
                "desktop UI with Scalene. The web UI shows training "
                "status in the bottom bar instead."
            )

    return components
