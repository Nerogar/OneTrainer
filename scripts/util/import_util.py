def script_imports(allow_zluda: bool = True):
    """
    Performs script imports for the project.

    This function handles setting up the correct library paths for
    the OneTrainer project and loads ZLUDA if applicable.
    It also filters out Triton warnings on startup.

    Args:
        allow_zluda (bool): Whether ZLUDA is allowed to be loaded.

    Returns:
        None: This function does not return a value.
    """
    import logging
    import os
    import sys
    from pathlib import Path

    # Filter out the Triton warning on startup.
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
