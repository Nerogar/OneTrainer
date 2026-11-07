# Training queue generator

You can open this by navigating to the Tools tab then clicking **Training Queue**. This tool is only available in Pyside UI as CTK is effectively defunct.

The generator writes a `.bat` or `.sh` containing ordinary
`python scripts/train.py --config-path ...` commands. Run that script to train.

- **Loop** points at existing exported training configs, in order and runs them verbatim.
- **Sweep** copies a base config for every combination of parameter values.
  The generated configs to a unique dir i.e `sweep-*` with distinct workspaces, caches, debug directories, model outputs,
  and cloud run IDs.

Sweep values accept comma lists, JSON lists, or inclusive numeric ranges such
as `0.01:0.03:0.01`. Use JSON strings for string parameters, for example
`["512", "1024"]` for `resolution`. Nested parameters such as
`optimizer.weight_decay` and `additional_embeddings[0].model_name` are supported.

Clicking `Generate Script` will output two files:
- training_queue.<bat/sh>
- training_queue.json which just enable reopening the generated file from the UI

The tool will not overwrite existing queues by default unless you enable it. Training uses the usual `secrets.json` unless specified otherwise

### Modifying secrets broadly

```console
training_queue.bat --secrets-path C:\credentials\alternate.json
```

Use an absolute secrets path, or one relative to the OneTrainer repository.
