import inspect
from collections.abc import Callable

from modules.util.tqdm_util import tqdm


def _accepted(process: Callable, available: dict) -> dict:
    # Pass a stage only the arguments its signature names (or everything, if it takes
    # **kwargs), so a stage's parameter list can be narrower than the context.
    params = inspect.signature(process).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return available
    return {name: value for name, value in available.items() if name in params}


def run_staged_pipeline(
        stages: list[tuple[str, Callable]],
        inputs: dict[str, list],
        shared: dict | None = None,
) -> list:
    # Run every item through stage 0, then every item through stage 1, and so on, instead
    # of running each item through all stages. Each stage is a (label, process) pair. A
    # sampler stage materializes the part it needs and evicts the rest, so this order
    # brings each part on-device once per stage rather than once per item.
    #
    # `inputs` is column-oriented - one list of per-item values per argument name -
    # transposed here into a context dict per item. Each context accumulates every
    # non-final stage's output, so a value produced early reaches any later stage without
    # being threaded through the ones between. The final stage's returns are collected
    # separately and returned. `shared` holds batch-level arguments offered to every
    # stage that names them.
    shared = shared or {}
    names = list(inputs)
    count = len(inputs[names[0]]) if names else 0
    contexts = [{name: inputs[name][i] for name in names} for i in range(count)]
    results = [None] * count
    last_index = len(stages) - 1
    for index, (label, process) in enumerate(stages):
        # One bar per stage, counting off the batch's items, so a stage with no inner
        # progress of its own is still visible while it runs. leave=False keeps finished
        # stages from piling up under the training bars; a stage with its own inner bar
        # nests below this one.
        for i, context in enumerate(tqdm(contexts, desc=label, leave=False)):
            output = process(**_accepted(process, {**shared, **context}))
            if index == last_index:
                results[i] = output
            else:
                context.update(output)
    # with no stages there is nothing to produce; hand back the raw contexts
    return results if stages else contexts
