import contextlib
from collections.abc import Callable

import huggingface_hub
from huggingface_hub import constants as hf_constants


def configure_hub(
        token: str = "",
        offline_mode: bool = False,
        cache_dir: str = "",
        on_status: Callable[[str], None] | None = None,
):
    # huggingface_hub reads HF_HUB_OFFLINE and HF_HUB_CACHE from the environment into module constants at import
    # time, and every downstream call reads those constants at call time. transformers and diffusers don't expose
    # a parameter on the internal requests they make (e.g. transformers queries model_info() while loading a
    # large-vocab tokenizer), so overriding the constants here is the only lever that reaches those calls and
    # redirects every HF request for the rest of the process.
    hf_constants.HF_HUB_OFFLINE = offline_mode
    if cache_dir:
        hf_constants.HF_HUB_CACHE = cache_dir

    if not offline_mode and token != "":
        if on_status is not None:
            on_status("logging into Hugging Face")
        with contextlib.suppress(ConnectionError):
            huggingface_hub.login(token=token)
