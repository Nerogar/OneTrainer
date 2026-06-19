#!/bin/bash
set -o pipefail

CONFIG="test/minimal.json"
LOG_DIR="test"
ERROR_LOG="$LOG_DIR/errors.log"

> "$ERROR_LOG"

presets=(
    "training_presets/#chroma LoRA 16GB.json"
    "training_presets/#ernie LoRA 16GB.json"
    "training_presets/#flux2 LoRA 16GB.json"
    "training_presets/#flux LoRA.json"
    "training_presets/#hidream LoRA.json"
    "training_presets/#hunyuan video LoRA.json"
    "training_presets/#pixart sigma 1.0 LoRA.json"
    "training_presets/#qwen LoRA 16GB.json"
    "training_presets/#sd 1.5 LoRA.json"
    "training_presets/#sd 2.1 LoRA.json"
    "training_presets/#sd 3 LoRA.json"
    "training_presets/#sdxl 1.0 LoRA.json"
    "training_presets/#wuerstchen 2.0 LoRA.json"
    "training_presets/#z-image DeTurbo LoRA 16GB.json"
    "training_presets/#z-image LoRA 16GB.json"
)

failed=0

for preset in "${presets[@]}"; do
    name=$(basename "$preset" .json)
    logfile="$LOG_DIR/${name#\#}.log"
    echo "=== $name ===" | tee "$logfile"
    if ./run-cmd.sh train --preset-path "$preset" --config-path "$CONFIG" 2>&1 | tee -a "$logfile"; then
        echo "OK: $name" | tee -a "$ERROR_LOG"
    else
        echo "FAILED: $name" | tee -a "$ERROR_LOG"
        failed=1
    fi
done

exit $failed
