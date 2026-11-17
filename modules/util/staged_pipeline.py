import inspect
from collections.abc import Callable

from modules.util.torch_util import torch_gc
from modules.util.tqdm_util import tqdm

import torch


def move_tensors(data, device: torch.device):
    """Copy tensor trees without retaining source views or changing their storage in place."""
    if isinstance(data, torch.Tensor):
        return data.detach().to(device=device)
    if isinstance(data, dict):
        return {key: move_tensors(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_tensors(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(move_tensors(value, device) for value in data)
    return data


def _argument_names(process: Callable) -> set[str] | None:
    params = inspect.signature(process).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def run_staged_pipeline(
        stages: list[tuple[str, Callable]],
        inputs: dict[str, list],
        shared: dict | None = None,
        *,
        on_result: Callable[[int, object], None] | None = None,
        on_error: Callable[[int, str, Exception], None] | None = None,
        offload_intermediates: bool = False,
) -> list:
    # Run every item through stage 0, then every item through stage 1, and so on, instead
    # of running each item through all stages. Each stage is a (label, process) pair. A
    # sampler stage materializes the part it needs and evicts the rest, so this order
    # brings each part on-device once per stage rather than once per item.
    #
    # `inputs` is column-oriented - one list of per-item values per argument name -
    # transposed here into a context dict per item. Retain values only while a later
    # stage needs them. With offloading, queued tensor trees live on CPU; each stage
    # restores its inputs after materializing its model part. Results remain aligned
    # with inputs, with None for failed items when on_error enables recovery.
    shared = shared or {}
    names = list(inputs)
    count = len(inputs[names[0]]) if names else 0
    if any(len(values) != count for values in inputs.values()):
        raise ValueError("Staged pipeline inputs must have equal lengths")
    contexts = [{name: inputs[name][i] for name in names} for i in range(count)]
    if not stages:
        return contexts

    argument_names = [_argument_names(process) for _, process in stages]
    future_names: list[set[str] | None] = [set() for _ in stages]
    for index in range(len(stages) - 2, -1, -1):
        following = future_names[index + 1]
        accepted = argument_names[index + 1]
        future_names[index] = None if following is None or accepted is None else following | accepted

    results = [None] * count
    last_index = len(stages) - 1
    for index, (label, process) in enumerate(stages):
        # One bar per stage, counting off the batch's items, so a stage with no inner
        # progress of its own is still visible while it runs. leave=False keeps finished
        # stages from piling up under the training bars; a stage with its own inner bar
        # nests below this one.
        for i, context in enumerate(tqdm(contexts, desc=label, leave=False)):
            if context is None:
                continue
            output = arguments = None
            failed = False
            error_stage = label
            try:
                accepted = argument_names[index]
                arguments = {**shared, **context}
                if accepted is not None:
                    arguments = {key: value for key, value in arguments.items() if key in accepted}
                output = process(**arguments)
                arguments = None
                if index == last_index:
                    context.clear()
                    error_stage = "output"
                    if on_result is not None:
                        on_result(i, output)
                    results[i] = output
                else:
                    context.update(output)
                    needed = future_names[index]
                    if needed is not None:
                        for key in list(context):
                            if key not in needed:
                                del context[key]
                    if offload_intermediates:
                        contexts[i] = move_tensors(context, torch.device("cpu"))
                        context.clear()
            except Exception as exc:
                context.clear()
                contexts[i] = None
                if on_error is None:
                    raise
                on_error(i, error_stage, exc)
                failed = True
            finally:
                # Do not let the previous call's arguments/output pin CUDA storage.
                output = arguments = None
            if failed:
                # The exception and its traceback have been released before collecting.
                torch_gc()
    return results
