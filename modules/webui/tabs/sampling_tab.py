"""Sampling settings tab for the Gradio WebUI.

Dynamic list of sample configurations, each editable inline.
"""

import gradio as gr

from modules.util.enum.NoiseScheduler import NoiseScheduler


def _default_sample() -> dict:
    """Return a default sample config as a plain dict."""
    return {
        "enabled": True,
        "prompt": "",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "frames": 1,
        "length": 1.0,
        "seed": 42,
        "random_seed": False,
        "cfg_scale": 7.0,
        "sampler": str(NoiseScheduler.DDIM),
        "steps": 20,
        "inpainting": False,
        "base_image_path": "",
        "mask_image_path": "",
        "sample_inpainting": False,
    }


def create_sampling_tab():
    """Build the 'sampling' tab using dynamic rendering."""
    components = {}

    sample_list = gr.State([])
    components["_sample_list_state"] = sample_list

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                components["sample_after"] = gr.Number(
                    label="Sample After",
                    value=30,
                    precision=0,
                    info="Create samples after a set time",
                    interactive=True,
                )
                components["sample_after_unit"] = gr.Dropdown(
                    label="Unit",
                    choices=["NEVER", "EPOCH", "STEP", "SECOND", "MINUTE", "HOUR", "ALWAYS"],
                    value="MINUTE",
                    interactive=True,
                )
        add_btn = gr.Button("Add Sample", variant="primary", scale=1)
        components["_sample_add_btn"] = add_btn

    @gr.render(inputs=[sample_list])
    def render_samples(samples):
        if not samples:
            gr.Markdown("*No samples configured. Click 'Add Sample' to begin.*")
            return

        for idx, sample in enumerate(samples):
            with gr.Group():
                with gr.Row():
                    gr.Checkbox(
                        label="Enabled",
                        value=sample.get("enabled", True),
                        scale=1,
                        interactive=True,
                    )
                    gr.Textbox(
                        label="Prompt",
                        value=sample.get("prompt", ""),
                        scale=4,
                        interactive=True,
                    )
                    remove_btn = gr.Button("🗑", variant="stop", scale=0, min_width=48)

                with gr.Accordion(f"Sample {idx + 1} — Settings", open=False):
                    gr.Textbox(
                        label="Negative Prompt",
                        value=sample.get("negative_prompt", ""),
                        interactive=True,
                    )
                    with gr.Row():
                        gr.Number(label="Width", value=sample.get("width", 512), precision=0, interactive=True)
                        gr.Number(label="Height", value=sample.get("height", 512), precision=0, interactive=True)
                        gr.Number(label="Frames", value=sample.get("frames", 1), precision=0, interactive=True)
                        gr.Number(label="Length", value=sample.get("length", 1.0), interactive=True)
                    with gr.Row():
                        gr.Number(label="Seed", value=sample.get("seed", 42), precision=0, interactive=True)
                        gr.Checkbox(label="Random Seed", value=sample.get("random_seed", False), interactive=True)
                        gr.Number(label="CFG Scale", value=sample.get("cfg_scale", 7.0), interactive=True)
                    with gr.Row():
                        gr.Dropdown(
                            label="Sampler",
                            choices=[str(x) for x in list(NoiseScheduler)],
                            value=sample.get("sampler", str(NoiseScheduler.DDIM)),
                            interactive=True,
                        )
                        gr.Number(label="Steps", value=sample.get("steps", 20), precision=0, interactive=True)
                    with gr.Row():
                        gr.Checkbox(label="Inpainting", value=sample.get("inpainting", False), interactive=True)
                        gr.Textbox(label="Base Image Path", value=sample.get("base_image_path", ""), interactive=True)
                        gr.Textbox(label="Mask Image Path", value=sample.get("mask_image_path", ""), interactive=True)

                def _remove_sample(samples_list, _idx=idx):
                    new_list = list(samples_list)
                    if 0 <= _idx < len(new_list):
                        new_list.pop(_idx)
                    return new_list

                remove_btn.click(
                    fn=_remove_sample,
                    inputs=[sample_list],
                    outputs=[sample_list],
                )

    def _add_sample(samples):
        return samples + [_default_sample()]

    add_btn.click(
        fn=_add_sample,
        inputs=[sample_list],
        outputs=[sample_list],
    )

    return components
