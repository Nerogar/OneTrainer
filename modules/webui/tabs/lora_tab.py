"""LoRA settings tab for the Gradio WebUI."""

import gradio as gr

from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType


def create_lora_tab():
    """Build the 'lora' tab and return a dict of components keyed by config attr."""
    components = {}

    with gr.Row():
        with gr.Column():
            components["lora_model_name"] = gr.Textbox(
                label="LoRA Model",
                value="",
                info="The LoRA base model. If empty, a new LoRA is created.",
                interactive=True,
            )
            components["lora_rank"] = gr.Number(
                label="LoRA Rank",
                value=16,
                precision=0,
                info="The rank of the LoRA model",
                interactive=True,
            )
            components["lora_alpha"] = gr.Number(
                label="LoRA Alpha",
                value=1.0,
                info="The alpha value of the LoRA model. Best set to be equal to or less than the rank",
                interactive=True,
            )
            components["peft_type"] = gr.Dropdown(
                label="Peft Type",
                choices=[str(x) for x in list(PeftType)],
                value=str(PeftType.LORA),
                info="The type of LoRA model to use",
                interactive=True,
            )
            components["dropout_probability"] = gr.Number(
                label="Dropout Probability",
                value=0.03,
                info="LoRA dropout probability",
                interactive=True,
            )
            components["oft_block_size"] = gr.Number(
                label="OFT Block Size",
                value=32,
                precision=0,
                info="The block size for Orthogonal Finetuning",
                interactive=True,
            )

        with gr.Column():
            components["lora_weight_dtype"] = gr.Dropdown(
                label="LoRA Weight Data Type",
                choices=[
                    ("float32", str(DataType.FLOAT_32)),
                    ("bfloat16", str(DataType.BFLOAT_16)),
                ],
                value=str(DataType.FLOAT_32),
                info="The weight data type of the LoRA",
                interactive=True,
            )
            components["bundle_additional_embeddings"] = gr.Checkbox(
                label="Bundle Embedded Embeddings",
                value=True,
                info="When enabled bundles the embeddings from the additional embeddings tab into the final LoRA output file",
                interactive=True,
            )

    return components
