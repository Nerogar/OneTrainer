def _patch_clip_text_model():
    # transformers 5.x flattened CLIPTextModel: .text_model no longer exists.
    # Add a property so older code (diffusers 0.38, OneTrainer) can still do
    # model.text_model.embeddings without breaking.
    try:
        from transformers.models.clip.modeling_clip import CLIPTextModel
        if not hasattr(CLIPTextModel, 'text_model'):
            CLIPTextModel.text_model = property(lambda self: self)
    except Exception:
        pass


def script_imports(allow_zluda: bool = True):
    import logging
    import os
    import sys
    from pathlib import Path

    _patch_clip_text_model()

    # Filter out the Triton warning on startup.
    # xformers is not installed anymore, but might still exist for some installations.
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # Insert ourselves as the highest-priority library path, so our modules are
    # always found without any risk of being shadowed by another import path.
    # 3 .parent calls to navigate from /scripts/util/import_util.py to the main directory
    onetrainer_lib_path = Path(__file__).absolute().parent.parent.parent
    sys.path.insert(0, str(onetrainer_lib_path))

    if allow_zluda and sys.platform.startswith('win'):
        from modules.zluda import ZLUDAInstaller

        zluda_path = ZLUDAInstaller.get_path()

        if os.path.exists(zluda_path):
            try:
                ZLUDAInstaller.load(zluda_path)
                print(f'Using ZLUDA in {zluda_path}')
            except Exception as e:
                print(f'Failed to load ZLUDA: {e}')

            from modules.zluda import ZLUDA

            ZLUDA.initialize()
