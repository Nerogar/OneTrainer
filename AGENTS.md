# AGENTS.md

OneTrainer trains diffusion models with full fine-tune / LoRA / embedding /
FINE_TUNE_VAE methods. Built on `diffusers` + PyTorch +
[`mgds`](https://github.com/Nerogar/MGDS). Supported families:
`modules/util/enum/ModelType.py`.

## Run

```
pre-commit install                            # install OUTSIDE the project venv
pre-commit run --all-files                    # mandatory before every PR
python scripts/train.py --config-path <preset.json> [--secrets-path secrets.json]
python scripts/train_ui.py                    # GUI (CustomTkinter)
```

Every `scripts/*.py` (except `generate_debug_report.py`) starts with `script_imports()`
from `scripts/util/import_util.py`: it fixes `sys.path`, filters xformers/Triton noise,
and bootstraps ZLUDA. Don't try to replicate it from a bare `python -c`.

There is **no automated test suite.** Verification is launching scripts and exercising
the change.

## Layout

```
modules/
  model/           BaseModel subclasses   — state (weights, optim, EMA, embeddings)
  modelLoader/     BaseModelLoader        — ckpt/safetensors/diffusers → model
  modelSetup/      BaseModelSetup         — optimizer, LR, grad checkpointing, device
  dataLoader/      BaseDataLoader         — wraps mgds dataset; bucketing, aug, captions
  modelSampler/    BaseModelSampler       — preview/inference sampling
  modelSaver/      BaseModelSaver         — model → disk
  trainer/         BaseTrainer, GenericTrainer (main loop), CloudTrainer (RunPod), MultiTrainer
  module/
    EMA / LoRA / OFT / AdditionalEmbedding training wrappers
    quantized/                            Fp8 / Nf4 / W8A8 / GGUFA8 / SVD linear layers
    Blip / ClipSeg / WD / Rembg / HPSv2 / AestheticScore  backends for caption_ui / generate_*
  ui/              CustomTkinter; one *UI/*Tab/*Window per screen.
                   TrainUI extends ctk.CTk; others extend ctk.CTkToplevel.
  util/
    create.py             ★ factory — TrainConfig → pipeline
    factory.py            auto-discovery (import_dir + class registry)
    config/               TrainConfig + sub-configs (Sample, Concept, Cloud, Secrets)
    enum/                 ModelType, Optimizer, LearningRateScheduler, NoiseScheduler,
                          TrainingMethod, DataType, ...
    callbacks/            TrainCallbacks (training-loop event hooks)
    commands/             TrainCommands (UI → trainer signals)
    optimizer/            CAME / adam / adamw / adafactor / muon extensions
    ui/                   UIState, components.py, validation.py
    LayerOffloadConductor.py  ⚠ fragile VRAM offloading; incompatible with dataloader_threads > 1
  zluda/           Windows AMD shim; disables cuDNN / flash_sdp / mem_efficient_sdp / cudnn_sdp

scripts/
  util/import_util.py  ★ script_imports() — sys.path + ZLUDA bootstrap; called first by every script
  train.py / train_ui.py / train_remote.py / sample.py
  caption_ui.py / convert_model_ui.py / video_tool_ui.py
  generate_captions.py / generate_masks.py / convert_model.py / calculate_loss.py
  create_train_files.py / install_zluda.py / generate_debug_report.py

training_presets/    ~50 partial-override JSONs of TrainConfig defaults, "#<model> <method>[ <vram>].json"
embedding_templates/ plain-text prompt files (`<embedding>` placeholder); .gitignore excludes user files
resources/           icons + sd_model_spec/*.json
docs/                ProjectStructure / Contributing / QuickStartGuide / CliTraining /
                     EmbeddingTraining / CaptioningAndMasking / RamOffloading / DockerImage
LAUNCH-SCRIPTS.md    OT_* env vars, venv/conda selection
lib.include.sh       runtime/venv/conda bootstrap; forces PYTORCH_ENABLE_MPS_FALLBACK=1 on macOS
```

## Architecture

### Factory + auto-discovery (`modules/util/create.py`)

`create_trainer(config, callbacks, commands)` is the top-level entry: returns
`CloudTrainer` if `config.cloud.enabled`, `MultiTrainer` if `config.multi_gpu`, else
`GenericTrainer` (the only branch that calls `ZLUDA.initialize_devices`).

`GenericTrainer.start()` then calls per-layer factories in order:
`create_model_loader → create_model_setup → create_data_loader → create_model_saver →
create_model_sampler`.

The selection mechanism for those five "per-model" layers is **auto-discovery**:

```python
factory.import_dir("modules/modelSampler", "modules.modelSampler")
factory.import_dir("modules/modelLoader",  "modules.modelLoader")
factory.import_dir("modules/modelSaver",   "modules.modelSaver")
factory.import_dir("modules/modelSetup",   "modules.modelSetup")
factory.import_dir("modules/dataLoader",   "modules.dataLoader")
```

Dropping a file with the right base class in any of those five directories registers
it — you do **not** edit `create.py` to register new per-model classes. You **do** edit
`create.py` for `create_optimizer`, `create_ema`, `create_lr_scheduler`,
`create_noise_scheduler`, `create_trainer` — those are explicit `match`/`case` branches
keyed off enums.

### UI

CustomTkinter. One file per screen in `modules/ui/`. `TrainUI` (the root window)
extends `ctk.CTk`; `CaptionUI`, `ConvertModelUI`, `VideoToolUI` extend `ctk.CTkToplevel`.
`TrainUI` composes tabs (`ModelTab`, `TrainingTab`, `SamplingTab`, `LoraTab`,
`ConceptTab`, `AdditionalEmbeddingsTab`, `CloudTab`); sub-windows are opened by whichever
component needs them.

Reactive state: `modules/util/ui/UIState.py` is `UIState(master, obj)` — it introspects
typed attributes on any config-shaped object and two-way-binds them to tkinter
vars. Shared widgets: `components.py`. Validation: `validation.py` +
`validation_helpers.py`.

### Config

`TrainConfig` (`modules/util/config/TrainConfig.py`) is **not** a `@dataclass`; it's a
`BaseConfig` subclass using class-level annotations. Serialization is via `to_dict()` /
`from_dict()` on `BaseConfig` — JSON conversion happens at the call site
(`scripts/train.py` does `json.load` → `from_dict`). `TrainConfig.py` also declares
several sub-configs inline (`TrainOptimizerConfig`, `TrainModelPartConfig`,
`TrainEmbeddingConfig`, `QuantizationConfig`); standalone sub-configs:
`SampleConfig.py`, `ConceptConfig.py`, `CloudConfig.py`, `SecretsConfig.py`.

Validation lives in `modules/util/ui/validation.py` — the **UI** layer. CLI scripts
bypass it. Validate at load time too if a field can be invalid from JSON.

Presets in `training_presets/` are partial overrides applied on top of
`TrainConfig.default_values()` (only fields that differ from defaults are present).

## Recipes

### Add a new model family

Pick the closest existing family as a template (`Flux2`, `Qwen`, `Sana`, `HiDream`,
`Chroma`, `HunyuanVideo`, `PixArtAlpha`, `StableDiffusion3`, etc.) — file-naming
conventions vary slightly per family, so copy the analog rather than guess.

1. Drop subclasses (auto-registered via `factory.import_dir`):
   - `modules/model/<Name>Model.py` ← `BaseModel`
   - `modules/modelLoader/<Name>{FineTune,LoRA,Embedding}ModelLoader.py` ← `BaseModelLoader`
     (some families instead use a single `<Name>ModelLoader.py`)
   - `modules/modelSetup/Base<Name>Setup.py` + per-method `<Name>{FineTune,LoRA,Embedding}Setup.py`
     ← `BaseModelSetup` (only the per-method classes auto-register)
   - `modules/dataLoader/<Name>BaseDataLoader.py` ← `BaseDataLoader`
   - `modules/modelSampler/<Name>Sampler.py` ← `BaseModelSampler` (one file per family)
   - `modules/modelSaver/<Name>{FineTune,LoRA,Embedding}ModelSaver.py` ← `BaseModelSaver`
2. Add `ModelType.<NAME>` to `modules/util/enum/ModelType.py`; mirror any sibling `is_<name>()` predicate.
3. Sweep `modules/ui/ModelTab.py` and `modules/ui/TrainingTab.py` for `if/elif model_type == ...` chains. Partial registration = silent feature absence.
4. Add a starter `training_presets/#<name> LoRA.json` (clone an analog's preset).
5. Optional: `resources/sd_model_spec/<name>.json`, `resources/icons/<name>.png`.

Cross-family shared logic lives in `modules/modelSetup/mixin/` and `modules/modelLoader/mixin/` — extend a mixin rather than duplicating.

### Add a new optimizer

1. `modules/util/enum/Optimizer.py` — add enum entry; update `is_adaptive` / `is_schedule_free` / `supports_fused_back_pass` predicates if applicable.
2. `modules/util/create.py::create_optimizer` — add `case Optimizer.<NAME>:`.
3. `modules/util/optimizer_util.py::OPTIMIZER_DEFAULT_PARAMETERS` — register default hyperparams.
4. If torch internals need patching: add a module under `modules/util/optimizer/` modelled on the existing `*_extensions.py`.
5. UI exposure: `modules/ui/OptimizerParamsWindow.py`.
6. Pin any new package in `requirements-global.txt` (or appropriate platform file).

### Add a new LR scheduler

1. `modules/util/enum/LearningRateScheduler.py` — add enum entry.
2. `modules/util/lr_scheduler_util.py` — add `lr_lambda_<name>(...)`.
3. `modules/util/create.py::create_lr_scheduler` — add `case` branch wiring the lambda.
4. UI exposure: `modules/ui/SchedulerParamsWindow.py` if it needs parameters.

### Add a new noise scheduler

1. `modules/util/enum/NoiseScheduler.py` — add enum entry.
2. `modules/util/create.py::create_noise_scheduler` — add `case` branch returning a `diffusers` scheduler. Mirror an existing case for the signature.
3. UI exposure: `modules/ui/SamplingTab.py`.

### Add a `TrainConfig` field

1. Declare a class-level annotation (`my_field: int`) on the appropriate `BaseConfig` subclass; set its default in that class's `default_values()`.
2. UI: bind via `UIState`. Add validation in `modules/util/ui/validation.py` if it can be invalid.
3. Presets keep loading (partial overrides). Migrate only if a default they relied on changes.
4. CLI bypasses UI validation — if a malformed JSON value would crash deeper in, validate at load.

### Hook into the training loop

- **Callbacks** (training → caller): `modules/util/callbacks/TrainCallbacks.py`. Instantiated in `scripts/train.py` / `TrainUI`, passed to `create_trainer(...)`.
- **Commands** (caller → training): `modules/util/commands/TrainCommands.py`. Available: `stop`, `sample_default`, `sample_custom(SampleConfig)`, `backup`, `save` (each paired with `get_and_reset_*` polled by the trainer). Multi-GPU sync: `modules/util/multi_gpu_util.py::sync_commands` + `TrainCommands.merge`.

## Footguns

- `modules/util/LayerOffloadConductor.py` — **incompatible with `dataloader_threads > 1`** when `gradient_checkpointing.offload()` and `layer_offload_fraction > 0` (hard-raised in `create.py`).
- ZLUDA (`modules/zluda/ZLUDA.py`) — `ZLUDA.initialize_devices(config)` runs only in the `GenericTrainer` branch of `create_trainer`. On Win-AMD it disables `cuDNN`, `flash_sdp`, `mem_efficient_sdp`, `cudnn_sdp`. Silent CPU fallback if its self-test fails.
- macOS — `lib.include.sh` exports `PYTORCH_ENABLE_MPS_FALLBACK=1`. Slow workflow on Mac? Suspect this first.
- `OT_*` env vars (`LAUNCH-SCRIPTS.md`) change install / venv selection / low-mem mode / platform requirements.
- CustomTkinter trace-removal workaround in `modules/util/ui/components.py` (references upstream CTK PR #2077, unlikely to merge). Don't simplify without re-verifying.
- `requirements-global.txt` has `git+https` commit pins for `diffusers`, `mgds`, `muon`. Don't bump silently.
- Adding a `ModelType` without sweeping every `if/elif model_type == ...` causes silent feature absence (no crash).
- CLI scripts bypass UI validation — a field validated only in `modules/util/ui/validation.py` can arrive malformed from JSON.

## Before opening a PR

- `pre-commit run --all-files` clean
- Launched the affected UI / script and exercised the change
- Touched the training path? Ran a short training job with a real preset
- No silent new top-level dependencies
- Can defend every line in the diff
