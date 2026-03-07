"""Additional Embeddings tab for the Gradio WebUI.

Dynamic list of embedding configurations, each editable inline.
"""

import gradio as gr

from modules.util.enum.TimeUnit import TimeUnit


def _default_embedding() -> dict:
    """Return a default embedding config as a plain dict."""
    return {
        "train": True,
        "model_name": "",
        "placeholder": "<embedding>",
        "token_count": 1,
        "initial_embedding_text": "*",
        "is_output_embedding": False,
        "stop_training_after": 0,
        "stop_training_after_unit": str(TimeUnit.NEVER),
    }


def create_embeddings_tab():
    """Build the 'additional embeddings' tab using dynamic rendering."""
    components = {}

    embedding_list = gr.State([])
    components["_embedding_list_state"] = embedding_list

    with gr.Row():
        add_btn = gr.Button("Add Embedding", variant="primary", scale=1)
        components["_embedding_add_btn"] = add_btn

    @gr.render(inputs=[embedding_list])
    def render_embeddings(embeddings):
        if not embeddings:
            gr.Markdown("*No additional embeddings. Click 'Add Embedding' to begin.*")
            return

        for idx, emb in enumerate(embeddings):
            with gr.Group():
                with gr.Row():
                    gr.Checkbox(
                        label="Train",
                        value=emb.get("train", True),
                        scale=1,
                        interactive=True,
                    )
                    gr.Textbox(
                        label="Model Name",
                        value=emb.get("model_name", ""),
                        scale=3,
                        interactive=True,
                    )
                    remove_btn = gr.Button("🗑", variant="stop", scale=0, min_width=48)

                with gr.Accordion(f"Embedding {idx + 1} — Settings", open=False):
                    with gr.Row():
                        gr.Textbox(
                            label="Placeholder",
                            value=emb.get("placeholder", "<embedding>"),
                            info="The placeholder token for this embedding",
                            interactive=True,
                        )
                        gr.Number(
                            label="Token Count",
                            value=emb.get("token_count", 1),
                            precision=0,
                            info="Number of tokens for this embedding",
                            interactive=True,
                        )
                    gr.Textbox(
                        label="Initial Embedding Text",
                        value=emb.get("initial_embedding_text", "*"),
                        info="Text used to initialize the embedding. '*' = random",
                        interactive=True,
                    )
                    gr.Checkbox(
                        label="Is Output Embedding",
                        value=emb.get("is_output_embedding", False),
                        interactive=True,
                    )
                    with gr.Row():
                        gr.Number(
                            label="Stop Training After",
                            value=emb.get("stop_training_after", 0),
                            precision=0,
                            interactive=True,
                        )
                        gr.Dropdown(
                            label="Unit",
                            choices=[str(x) for x in list(TimeUnit)],
                            value=emb.get("stop_training_after_unit", str(TimeUnit.NEVER)),
                            interactive=True,
                        )

                def _remove_embedding(emb_list, _idx=idx):
                    new_list = list(emb_list)
                    if 0 <= _idx < len(new_list):
                        new_list.pop(_idx)
                    return new_list

                remove_btn.click(
                    fn=_remove_embedding,
                    inputs=[embedding_list],
                    outputs=[embedding_list],
                )

    def _add_embedding(embeddings):
        return embeddings + [_default_embedding()]

    add_btn.click(
        fn=_add_embedding,
        inputs=[embedding_list],
        outputs=[embedding_list],
    )

    return components
