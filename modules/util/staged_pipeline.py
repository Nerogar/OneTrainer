import inspect
from collections.abc import Callable

from modules.util.tqdm_util import tqdm


def _accepted(process: Callable, available: dict) -> dict:
    # Pass a stage only the arguments its signature names (or everything, if it
    # takes **kwargs). This keeps each stage's parameter list limited to what it
    # actually consumes, even though the context carries more.
    params = inspect.signature(process).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return available
    return {name: value for name, value in available.items() if name in params}


def run_staged_pipeline(
        stages: list[tuple[str, Callable]],
        inputs: dict[str, list],
        shared: dict | None = None,
) -> list:
    # Run every item through stage 0, then every item through stage 1, and so on
    # (stage-major order), instead of running each item through all stages before
    # starting the next (item-major order). Each stage is a (label, process) pair;
    # `label` names the stage on its progress bar, `process` is a callable run
    # once per item. Stage-major order is what bounds how often a model part has to
    # be brought on-device. A sampler stage materializes the part it needs and evicts
    # the rest, so item-major would cycle text encoder -> transformer -> vae once per
    # item, moving all three onto the train device every time; stage-major moves each
    # one there once and keeps it for the stage's whole run. What that saves scales
    # with how an evicted part is held - a host-to-device copy of weights already in
    # RAM, or a re-read of the weights from disk once parts are streamed.
    #
    # `inputs` is column-oriented: each key maps to the list of that argument's
    # per-item values (all lists the same length), transposed here into one context
    # dict per item. Each context then accumulates every non-final stage's output,
    # so a value produced (or supplied) early is available to any later stage
    # without being threaded through the ones in between. The final stage's return
    # value is collected into a separate result list - one entry per item - and that
    # list is what this function returns, so a pipeline outputs whatever its last
    # stage returns. `shared` holds batch-level arguments (e.g. a progress reporter)
    # offered to every stage that names them.
    shared = shared or {}
    names = list(inputs)
    count = len(inputs[names[0]]) if names else 0
    contexts = [{name: inputs[name][i] for name in names} for i in range(count)]
    results = [None] * count
    last_index = len(stages) - 1
    for index, (label, process) in enumerate(stages):
        # One bar per stage, counting off the batch's items, so a stage that shows no
        # inner progress of its own (text encoding, VAE decoding) is still visible while
        # it runs. leave=False keeps the finished stages from piling up in the log under
        # the training bars. A stage with its own inner bar (denoising counts diffusion
        # steps) nests below this one; tqdm assigns the positions itself.
        for i, context in enumerate(tqdm(contexts, desc=label, leave=False)):
            output = process(**_accepted(process, {**shared, **context}))
            if index == last_index:
                results[i] = output
            else:
                context.update(output)
    # with no stages there is nothing to produce; hand back the raw contexts
    return results if stages else contexts
