def script_imports(allow_zluda: bool = True):
    import logging
    import os
    import re
    import sys
    import warnings
    from pathlib import Path

    # Suppress the Triton startup warning and leftover xformers installations warnings.
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # Silence non-actionable startup/compile warnings unless OT_DEBUG_WARNINGS is set.
    # Filter emitting loggers directly; parent filters miss child records.
    if not os.environ.get("OT_DEBUG_WARNINGS"):
        # Suppress noisy diffusers/transformers import/load logs; regexes match runtime values.
        logging.getLogger("diffusers.configuration_utils").addFilter(
            lambda record: not re.search(
                r"The config attributes .* were passed to .*, but are not expected and will be ignored",
                record.getMessage(),
            )
        )
        logging.getLogger("diffusers.models.modeling_utils").addFilter(
            lambda record: 'Attention backends are an experimental feature' not in record.getMessage()
        )
        logging.getLogger("transformers.modeling_utils").addFilter(
            lambda record: not re.search(
                r"`loss_type=.*` was set in the config but it is unrecognized", record.getMessage()
            )
        )

        # A dependency still passes the deprecated local_dir_use_symlinks to hf_hub_download.
        warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")

        # Suppress harmless torch.compile fallback notes: complex operators use warnings.warn(),
        # while insufficient SMs use logger.warning().
        warnings.filterwarnings("ignore", message=r".*does not support code generation for complex operators.*")
        logging.getLogger("torch._inductor.utils").addFilter(
            lambda record: 'Not enough SMs to use max_autotune_gemm mode' not in record.getMessage()
        )

    # Prioritize local modules to prevent shadowing; three parents reach the repo root.
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
