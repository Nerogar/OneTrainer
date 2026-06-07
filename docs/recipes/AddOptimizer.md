# Adding a New Optimizer

1. `modules/util/enum/Optimizer.py` — add enum entry; update `is_adaptive` / `is_schedule_free` / `supports_fused_back_pass` predicates if applicable.
2. `modules/util/create.py::create_optimizer` — add `case Optimizer.<NAME>:`.
3. `modules/util/optimizer_util.py::OPTIMIZER_DEFAULT_PARAMETERS` — register default hyperparams.
4. If torch internals need patching: add a module under `modules/util/optimizer/` modelled on the existing `*_extensions.py`.
5. UI exposure: `modules/ui/OptimizerParamsWindow.py`.
6. Pin any new package in `requirements-global.txt` (or appropriate platform file).
