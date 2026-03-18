"""Data settings tab for the Gradio WebUI."""

import gradio as gr


def create_data_tab():
    """Build the 'data' tab and return a dict of components keyed by config attr."""
    components = {}

    with gr.Row():
        with gr.Column():
            components["concept_file_name"] = gr.Textbox(
                label="Concept File Name",
                value="training_concepts/concepts.json",
                info="The json file containing the concept definitions",
                interactive=True,
            )
            components["circular_mask_generation"] = gr.Checkbox(
                label="Circular Mask Generation",
                value=False,
                info="Automatically create circular masks for masked training",
                interactive=True,
            )
            components["random_rotate_and_crop"] = gr.Checkbox(
                label="Random Rotate and Crop",
                value=False,
                info="Randomly rotate the training samples and crop to the masked region",
                interactive=True,
            )

        with gr.Column():
            components["aspect_ratio_bucketing"] = gr.Checkbox(
                label="Aspect Ratio Bucketing",
                value=True,
                info="Enables aspect ratio bucketing",
                interactive=True,
            )
            components["latent_caching"] = gr.Checkbox(
                label="Latent Caching",
                value=True,
                info="Enables latent caching to speed up training",
                interactive=True,
            )
            components["clear_cache_before_training"] = gr.Checkbox(
                label="Clear Cache Before Training",
                value=False,
                info="Clears the latent cache before starting the training run",
                interactive=True,
            )

    return components
