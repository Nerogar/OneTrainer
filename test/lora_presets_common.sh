CONFIG="test/minimal.json"
LOG_DIR="test"
ERROR_LOG="$LOG_DIR/errors.log"
LORA_DIR="models/lora_smoke"

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

# ComfyUI only supports Stable Cascade, an architecturally different model from
# the Wuerstchen v2 that OT trains, so there is no Comfy smoke test for it.
comfy_unsupported=("WUERSTCHEN_2")

preset_name() {
    basename "$1" .json
}

preset_logfile() {
    local name
    name=$(preset_name "$1")
    echo "$LOG_DIR/${name#\#}.log"
}

preset_lora_path() {
    local name
    name=$(preset_name "$1")
    echo "$LORA_DIR/${name#\#}/model.safetensors"
}
