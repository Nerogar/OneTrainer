"""Model settings tab for the Gradio WebUI.

Contains all model-type-conditional fields: base model path, dtype selectors
for UNet/Transformer/Prior/Text Encoders/VAE, quantization, and output settings.
Sections show/hide based on the selected model_type via .change() callbacks.
"""

import gradio as gr

from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType


# ── helpers ─────────────────────────────────────────────────────────────────

def _dtype_choices(include_gguf=False, include_a8=False):
    """Return dtype choice tuples matching the CTk options_kv pattern."""
    choices = [
        ("float32", str(DataType.FLOAT_32)),
        ("bfloat16", str(DataType.BFLOAT_16)),
        ("float16", str(DataType.FLOAT_16)),
        ("float8 (W8)", str(DataType.FLOAT_8)),
        ("nfloat4", str(DataType.NFLOAT_4)),
    ]
    if include_a8:
        choices += [
            ("float W8A8", str(DataType.FLOAT_W8A8)),
            ("int W8A8", str(DataType.INT_W8A8)),
        ]
    if include_gguf:
        choices += [
            ("GGUF", str(DataType.GGUF)),
            ("GGUF A8 float", str(DataType.GGUF_A8_FLOAT)),
            ("GGUF A8 int", str(DataType.GGUF_A8_INT)),
        ]
    return choices


def _model_type_flags(model_type_str: str) -> dict:
    """Compute which sections are visible for a given model type string."""
    mt = model_type_str
    return {
        "has_unet": mt in [
            str(ModelType.STABLE_DIFFUSION_15),
            str(ModelType.STABLE_DIFFUSION_15_INPAINTING),
            str(ModelType.STABLE_DIFFUSION_XL),
            str(ModelType.STABLE_DIFFUSION_XL_INPAINTING),
        ],
        "has_prior": "WUERSTCHEN" in mt or "STABLE_CASCADE" in mt,
        "has_transformer": mt not in [
            str(ModelType.STABLE_DIFFUSION_15),
            str(ModelType.STABLE_DIFFUSION_15_INPAINTING),
            str(ModelType.STABLE_DIFFUSION_XL),
            str(ModelType.STABLE_DIFFUSION_XL_INPAINTING),
        ] and "WUERSTCHEN" not in mt and "STABLE_CASCADE" not in mt,
        "has_single_te": mt in [
            str(ModelType.STABLE_DIFFUSION_15),
            str(ModelType.STABLE_DIFFUSION_15_INPAINTING),
            str(ModelType.PIXART_ALPHA),
            str(ModelType.PIXART_SIGMA),
            str(ModelType.SANA),
            str(ModelType.FLUX_DEV_2),
            str(ModelType.CHROMA),
            str(ModelType.QWEN2),
            str(ModelType.ZIMAGEN),
        ] or "WUERSTCHEN" in mt or "STABLE_CASCADE" in mt,
        "has_te_1": "STABLE_DIFFUSION_XL" in mt or "STABLE_DIFFUSION_3" in mt
                    or "FLUX" in mt or "HUNYUAN" in mt or "HIDREAM" in mt,
        "has_te_2": "STABLE_DIFFUSION_XL" in mt or "STABLE_DIFFUSION_3" in mt
                    or "FLUX" in mt or "HUNYUAN" in mt or "HIDREAM" in mt,
        "has_te_3": "STABLE_DIFFUSION_3" in mt or "HIDREAM" in mt,
        "has_te_4": "HIDREAM" in mt,
        "has_vae": "WUERSTCHEN" not in mt and "STABLE_CASCADE" not in mt,
        "has_effnet": "WUERSTCHEN" in mt or "STABLE_CASCADE" in mt,
        "has_decoder": "WUERSTCHEN" in mt or "STABLE_CASCADE" in mt,
    }


# ── tab builder ─────────────────────────────────────────────────────────────

def create_model_tab():
    """Build the 'model' tab and return (components_dict, setup_fn)."""
    components = {}

    # ── always visible: base ────────────────────────────────────────
    components["secrets.huggingface_token"] = gr.Textbox(
        label="Hugging Face Token",
        value="",
        type="password",
        info="Enter your Hugging Face access token for protected repositories",
        interactive=True,
    )

    with gr.Row():
        components["base_model_name"] = gr.Textbox(
            label="Base Model",
            value="stable-diffusion-v1-5/stable-diffusion-v1-5",
            info="Filename, directory or Hugging Face repository of the base model",
            scale=4,
            interactive=True,
        )
        components["compile"] = gr.Checkbox(
            label="Compile transformer blocks",
            value=False,
            info="Uses torch.compile and Triton to speed up training",
            scale=1,
            interactive=True,
        )

    # ── conditional: UNet dtype ─────────────────────────────────────
    with gr.Group(visible=True) as unet_group:
        components["unet.weight_dtype"] = gr.Dropdown(
            label="UNet Data Type",
            choices=_dtype_choices(include_a8=True),
            value=str(DataType.FLOAT_32),
            info="The unet weight data type",
            interactive=True,
        )
    components["_unet_group"] = unet_group

    # ── conditional: Prior ──────────────────────────────────────────
    with gr.Group(visible=False) as prior_group:
        components["prior.model_name"] = gr.Textbox(
            label="Prior Model",
            value="",
            info="Filename, directory or Hugging Face repository of the prior model",
            interactive=True,
        )
        components["prior.weight_dtype"] = gr.Dropdown(
            label="Prior Data Type",
            choices=_dtype_choices(),
            value=str(DataType.FLOAT_32),
            info="The prior weight data type",
            interactive=True,
        )
    components["_prior_group"] = prior_group

    # ── conditional: Transformer ────────────────────────────────────
    with gr.Group(visible=False) as transformer_group:
        components["transformer.model_name"] = gr.Textbox(
            label="Override Transformer / GGUF",
            value="",
            info="Can be used to override the transformer. Safetensors and GGUF files supported.",
            interactive=True,
        )
        components["transformer.weight_dtype"] = gr.Dropdown(
            label="Transformer Data Type",
            choices=_dtype_choices(include_gguf=True, include_a8=True),
            value=str(DataType.FLOAT_32),
            info="The transformer weight data type",
            interactive=True,
        )
    components["_transformer_group"] = transformer_group

    # ── quantization (always visible) ───────────────────────────────
    with gr.Accordion("Quantization", open=False):
        with gr.Row():
            components["quantization.layer_filter_preset"] = gr.Dropdown(
                label="Quantization Layer Filter Preset",
                choices=["full"],
                value="full",
                interactive=True,
            )
            components["quantization.layer_filter"] = gr.Textbox(
                label="Layer Filter",
                value="",
                info="Comma-separated layer names for quantization",
                interactive=True,
            )
            components["quantization.layer_filter_regex"] = gr.Checkbox(
                label="Regex",
                value=False,
                interactive=True,
            )
        with gr.Row():
            components["quantization.svd_dtype"] = gr.Dropdown(
                label="SVDQuant",
                choices=[
                    ("disabled", str(DataType.NONE)),
                    ("float32", str(DataType.FLOAT_32)),
                    ("bfloat16", str(DataType.BFLOAT_16)),
                ],
                value=str(DataType.NONE),
                info="What datatype to use for SVDQuant weights decomposition",
                interactive=True,
            )
            components["quantization.svd_rank"] = gr.Number(
                label="SVDQuant Rank",
                value=32,
                precision=0,
                info="Rank for SVDQuant weights decomposition",
                interactive=True,
            )

    # ── conditional: single text encoder ────────────────────────────
    with gr.Group(visible=True) as te_group:
        components["text_encoder.weight_dtype"] = gr.Dropdown(
            label="Text Encoder Data Type",
            choices=_dtype_choices(),
            value=str(DataType.FLOAT_32),
            info="The text encoder weight data type",
            interactive=True,
        )
    components["_te_group"] = te_group

    # ── conditional: numbered text encoders 1-4 ─────────────────────
    for i, suffix in [(1, ""), (2, "_2"), (3, "_3"), (4, "_4")]:
        vis = False  # initially hidden; model_type.change sets visibility
        with gr.Group(visible=vis) as te_n_group:
            attr = f"text_encoder{suffix}"
            components[f"{attr}.weight_dtype"] = gr.Dropdown(
                label=f"Text Encoder {i} Data Type",
                choices=_dtype_choices(),
                value=str(DataType.FLOAT_32),
                info=f"The text encoder {i} weight data type",
                interactive=True,
            )
        components[f"_te{i}_group"] = te_n_group

    # ── conditional: VAE ────────────────────────────────────────────
    with gr.Group(visible=True) as vae_group:
        with gr.Row():
            components["vae.model_name"] = gr.Textbox(
                label="VAE Override",
                value="",
                info="Directory or HF repo of a VAE model in diffusers format",
                interactive=True,
            )
            components["vae.weight_dtype"] = gr.Dropdown(
                label="VAE Data Type",
                choices=_dtype_choices(),
                value=str(DataType.FLOAT_32),
                info="The vae weight data type",
                interactive=True,
            )
    components["_vae_group"] = vae_group

    # ── conditional: Wuerstchen effnet + decoder ────────────────────
    with gr.Group(visible=False) as effnet_group:
        with gr.Row():
            components["effnet_encoder.model_name"] = gr.Textbox(
                label="Effnet Encoder Model",
                value="",
                interactive=True,
            )
            components["effnet_encoder.weight_dtype"] = gr.Dropdown(
                label="Effnet Encoder Data Type",
                choices=_dtype_choices(),
                value=str(DataType.FLOAT_32),
                interactive=True,
            )
        with gr.Row():
            components["decoder.model_name"] = gr.Textbox(
                label="Decoder Model",
                value="",
                interactive=True,
            )
            components["decoder.weight_dtype"] = gr.Dropdown(
                label="Decoder Data Type",
                choices=_dtype_choices(),
                value=str(DataType.FLOAT_32),
                interactive=True,
            )
    components["_effnet_group"] = effnet_group

    # ── output (always visible) ─────────────────────────────────────
    gr.Markdown("### Output")
    with gr.Row():
        components["output_model_destination"] = gr.Textbox(
            label="Model Output Destination",
            value="models/model.safetensors",
            info="Filename or directory where the output model is saved",
            scale=3,
            interactive=True,
        )
        components["output_dtype"] = gr.Dropdown(
            label="Output Data Type",
            choices=[
                ("float16", str(DataType.FLOAT_16)),
                ("float32", str(DataType.FLOAT_32)),
                ("bfloat16", str(DataType.BFLOAT_16)),
                ("float8", str(DataType.FLOAT_8)),
                ("nfloat4", str(DataType.NFLOAT_4)),
            ],
            value=str(DataType.FLOAT_32),
            info="Precision to use when saving the output model",
            scale=1,
            interactive=True,
        )

    with gr.Row():
        components["output_model_format"] = gr.Dropdown(
            label="Output Format",
            choices=[
                ("Safetensors", str(ModelFormat.SAFETENSORS)),
                ("Diffusers", str(ModelFormat.DIFFUSERS)),
            ],
            value=str(ModelFormat.SAFETENSORS),
            info="Format to use when saving the output model",
            interactive=True,
        )
        components["include_train_config"] = gr.Dropdown(
            label="Include Config",
            choices=[
                ("None", str(ConfigPart.NONE)),
                ("Settings", str(ConfigPart.SETTINGS)),
                ("All", str(ConfigPart.ALL)),
            ],
            value=str(ConfigPart.NONE),
            info="Include the training configuration in the final model",
            interactive=True,
        )

    return components


def update_model_tab_visibility(model_type_str: str, components: dict):
    """
    Called from model_type.change() to show/hide conditional sections.
    Returns a dict of gr.update() calls keyed by the group component refs.
    """
    flags = _model_type_flags(model_type_str)
    return {
        components["_unet_group"]: gr.update(visible=flags["has_unet"]),
        components["_prior_group"]: gr.update(visible=flags["has_prior"]),
        components["_transformer_group"]: gr.update(visible=flags["has_transformer"]),
        components["_te_group"]: gr.update(visible=flags["has_single_te"]),
        components["_te1_group"]: gr.update(visible=flags["has_te_1"]),
        components["_te2_group"]: gr.update(visible=flags["has_te_2"]),
        components["_te3_group"]: gr.update(visible=flags["has_te_3"]),
        components["_te4_group"]: gr.update(visible=flags["has_te_4"]),
        components["_vae_group"]: gr.update(visible=flags["has_vae"]),
        components["_effnet_group"]: gr.update(visible=flags["has_effnet"]),
    }
