def script_imports(allow_zluda: bool = True):
    import logging
    import os
    import re
    import sys
    import warnings
    from pathlib import Path

    # Filter out the Triton warning on startup.
    # xformers is not installed anymore, but might still exist for some installations.
    logging \
        .getLogger("xformers") \
        .addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

    # Silence specific non-actionable startup/compile warnings. A logger filter
    # targets the exact emitting logger, since a parent logger's filter misses
    # records from child loggers.

    # diffusers/transformers chatty logger.warning() lines at import/load time.
    logging.getLogger("diffusers.modular_pipelines").addFilter(
        lambda record: 'Modular Diffusers is currently an experimental feature' not in record.getMessage()
    )
    # The subject of these two is interpolated into the message, so match the whole
    # sentence with .* standing in for the runtime value.
    logging.getLogger("diffusers.configuration_utils").addFilter(
        lambda record: not re.search(
            r"The config attributes .* were passed to .*, but are not expected and will be ignored",
            record.getMessage(),
        )
    )
    logging.getLogger("transformers.modeling_utils").addFilter(
        lambda record: not re.search(
            r"`loss_type=.*` was set in the config but it is unrecognized", record.getMessage()
        )
    )

    # A dependency still calls hf_hub_download with the removed local_dir_use_symlinks
    # argument; the deprecation warning is not actionable.
    warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")

    # torch.compile emits performance notes when inductor falls back or can't use a
    # fast path; harmless and noisy for normal runs. The SMs note is a logger.warning()
    # on its exact emitting logger; the complex-operators note is a warnings.warn().
    warnings.filterwarnings("ignore", message=r".*does not support code generation for complex operators.*")
    logging.getLogger("torch._inductor.utils").addFilter(
        lambda record: 'Not enough SMs to use max_autotune_gemm mode' not in record.getMessage()
    )

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
