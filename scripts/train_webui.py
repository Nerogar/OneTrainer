from util.import_util import script_imports

script_imports()

import gradio as gr

from modules.webui.app import create_app, LOGO_DIR


def main():
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
        allowed_paths=[LOGO_DIR],
    )


if __name__ == '__main__':
    main()
