def _patch_clip_text_model():
    # transformers 5.6 flattened CLIPTextModel (removed the .text_model wrapper).
    # diffusers 0.38+ was updated for the flat layout, so it accesses
    # embeddings/encoder/final_layer_norm directly on the model.
    #
    # - transformers >=5.6 (flat):  add text_model = self  so old OneTrainer code works
    # - transformers <=5.5 (nested): expose embeddings/encoder/final_layer_norm
    #   directly via _modules['text_model'] so diffusers 0.38 code works
    try:
        import transformers as _tr
        _major, _minor = (int(x) for x in _tr.__version__.split('.')[:2])
        from transformers.models.clip.modeling_clip import CLIPTextModel

        if (_major, _minor) >= (5, 6):
            # flat layout — text_model no longer exists; point it back to self
            if not hasattr(CLIPTextModel, 'text_model'):
                CLIPTextModel.text_model = property(lambda self: self)
        else:
            # nested layout — diffusers 0.38 tries to access flat attr names
            def _nested(attr):
                return property(lambda self: getattr(self._modules['text_model'], attr))
            for _attr in ('embeddings', 'encoder', 'final_layer_norm'):
                if not hasattr(CLIPTextModel, _attr):
                    setattr(CLIPTextModel, _attr, _nested(_attr))
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
