from util.import_util import script_imports
script_imports()
import argparse
import gradio as gr
from modules.webui.app import create_app, LOGO_DIR

def main():
    parser = argparse.ArgumentParser(description="Launch OneTrainer WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--root-path", default="", dest="root_path")
    parser.add_argument("--share", action="store_true", default=False)
    
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        theme=gr.themes.Soft(),
        allowed_paths=[LOGO_DIR],
    )

if __name__ == '__main__':
    main()
