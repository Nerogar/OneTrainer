"""Cloud settings tab for the Gradio WebUI."""

import gradio as gr


def create_cloud_tab():
    """Build the 'cloud' tab and return a dict of components keyed by config attr."""
    components = {}

    with gr.Row():
        with gr.Column():
            components["cloud.cloud_provider"] = gr.Dropdown(
                label="Cloud Provider",
                choices=["NONE", "RUNPOD"],
                value="NONE",
                info="The cloud provider to use for training",
                interactive=True,
            )
            components["secrets.cloud_api_key"] = gr.Textbox(
                label="API Key",
                value="",
                type="password",
                info="The API key for the cloud provider",
                interactive=True,
            )
            components["cloud.sync_files"] = gr.Textbox(
                label="Sync Files",
                value="",
                info="Files and directories to sync to the cloud",
                interactive=True,
            )
            components["cloud.gpu_type"] = gr.Textbox(
                label="GPU Type",
                value="",
                info="The type of GPU to use",
                interactive=True,
            )
            components["cloud.gpu_count"] = gr.Number(
                label="GPU Count",
                value=1,
                precision=0,
                info="The number of GPUs to use",
                interactive=True,
            )

        with gr.Column():
            components["cloud.remote_dir"] = gr.Textbox(
                label="Remote Directory",
                value="/workspace",
                info="The remote directory for training",
                interactive=True,
            )
            components["cloud.install_cmd"] = gr.Textbox(
                label="Install Command",
                value="",
                info="Extra command to run during cloud setup",
                interactive=True,
            )
            components["cloud.on_training_complete"] = gr.Dropdown(
                label="On Training Complete",
                choices=["NOTHING", "STOP_MACHINE", "TERMINATE_MACHINE"],
                value="NOTHING",
                info="Action to take when training completes",
                interactive=True,
            )
            components["cloud.on_training_error"] = gr.Dropdown(
                label="On Training Error",
                choices=["NOTHING", "STOP_MACHINE", "TERMINATE_MACHINE"],
                value="NOTHING",
                info="Action to take when training encounters an error",
                interactive=True,
            )

    return components
