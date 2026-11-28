# SDXL long captions

Set **Caption Tokens** below **Resolution** in the SDXL Training tab. Choose 75 (the default), 150, or 225 content tokens. Training previews inherit this setting.

- Headless training uses `--config-value sdxl_max_caption_tokens=150`.
- Standalone sampling uses `--sdxl-max-caption-tokens 150`.

Each additional 75-token chunk adds up to 77 cross-attention positions. Extended capacities use separate text caches, and the default remains compatible with existing 75-token caches.
