def check_lora_rank_alpha_match(model, config):
    """
    Checks if the LoRA rank in the state dict matches the config.
    Raises a ValueError if mismatch.
    """
    if not model.lora_state_dict:
        return

    rank_from_state = None
    for key, tensor in model.lora_state_dict.items():
        if 'lora_down' in key and 'weight' in key:
            rank_from_state = min(tensor.shape)
            break

    if rank_from_state is not None and rank_from_state != config.lora_rank:
        raise ValueError(
            f"LoRA Rank and Alpha settings must match what the original LoRA was trained with. "
            f"Config/UI specifies rank {config.lora_rank}, but the loaded LoRA state dict was trained with a rank of {rank_from_state}. "
            f"Please update your LoRA rank in the UI to match."
        )

    alpha_from_state = None
    for key, tensor in model.lora_state_dict.items():
        if '.alpha' in key:
            alpha_from_state = tensor.item()
            break

    if alpha_from_state is not None and alpha_from_state != config.lora_alpha:
        raise ValueError(
            f"LoRA Alpha mismatch: Config/UI specifies alpha {config.lora_alpha}, but state dict has {alpha_from_state}. "
            f"LoRA Rank and Alpha settings must match what the original LoRA was trained with."
        )
