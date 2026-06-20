import argparse
import os

from huggingface_hub import hf_hub_download

# Comfy-format files needed per OT model_type to run smoke_load_comfy.py.
# First entry under "files" is always the unet/checkpoint; any further entries
# are text-encoder files, in the order smoke_load_comfy.py expects them.
# Sources verified against the actual HF repo file listings and, where available,
# against the "Model Links" MarkdownNote baked into ComfyUI's own workflow templates
# (comfyui_workflow_templates_media_image/templates/*.json) - not guessed from memory.
MODELS = {
    "CHROMA_1": {
        "files": [
            ("Comfy-Org/Chroma1-HD_repackaged", "split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors"),
            ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors"),
        ],
    },
    "ERNIE": {
        "files": [
            ("Comfy-Org/ERNIE-Image", "diffusion_models/ernie-image.safetensors"),
            ("Comfy-Org/ERNIE-Image", "text_encoders/ministral-3-3b.safetensors"),
        ],
    },
    "FLUX_DEV_1": {
        "files": [
            ("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
            ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
            ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors"),
        ],
    },
    "FLUX_2": {
        "files": [
            ("black-forest-labs/FLUX.2-klein-base-9b-fp8", "flux-2-klein-base-9b-fp8.safetensors"),
            ("Comfy-Org/flux2-klein-9B", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"),
        ],
    },
    "HI_DREAM_FULL": {
        "files": [
            ("Comfy-Org/HiDream-I1_ComfyUI", "split_files/diffusion_models/hidream_i1_full_fp8.safetensors"),
            ("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/clip_l_hidream.safetensors"),
            ("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/clip_g_hidream.safetensors"),
            ("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors"),
            ("Comfy-Org/HiDream-I1_ComfyUI", "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors"),
        ],
    },
    "HUNYUAN_VIDEO": {
        "files": [
            ("Comfy-Org/HunyuanVideo_repackaged", "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors"),
            ("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/clip_l.safetensors"),
            ("Comfy-Org/HunyuanVideo_repackaged", "split_files/text_encoders/llava_llama3_fp8_scaled.safetensors"),
        ],
    },
    "PIXART_SIGMA": {
        "files": [
            ("HDiffusion/Pixart-Sigma-Safetensors", "PixArt-Sigma-XL-2-1024-MS.safetensors"),
            ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
        ],
    },
    "QWEN": {
        "files": [
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"),
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        ],
    },
    "STABLE_DIFFUSION_15": {
        "files": [
            ("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
        ],
    },
    "STABLE_DIFFUSION_21": {
        "files": [
            ("sd2-community/stable-diffusion-2-1", "v2-1_768-ema-pruned.safetensors"),
        ],
    },
    "STABLE_DIFFUSION_3": {
        "files": [
            ("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp8.safetensors"),
        ],
    },
    "STABLE_DIFFUSION_XL_10_BASE": {
        "files": [
            ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
        ],
    },
    "Z_IMAGE": {
        "files": [
            ("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"),
        ],
    },
    # WUERSTCHEN_2 is intentionally omitted: ComfyUI only supports Stable Cascade,
    # an architecturally different model from the Wuerstchen v2 that OT trains.
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=MODELS.keys())
    parser.add_argument("--dest-dir", default="models/comfy_smoke")
    args = parser.parse_args()

    files = MODELS[args.model_type]["files"]
    dest_dir = os.path.join(args.dest_dir, args.model_type)
    os.makedirs(dest_dir, exist_ok=True)

    for repo_id, filename in files:
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
        print(path)


if __name__ == '__main__':
    main()
