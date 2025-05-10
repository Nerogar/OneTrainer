
# --- SageAttention Import --- #
try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False
    print("Warning: SageAttention not found, SageAttentionProcessor will not be functional.")
    sageattn = None # Define sageattn as None if not available

# --- Refactored SageAttention Processor (SD/SDXL) --- #
if SAGE_ATTENTION_AVAILABLE:
    class SageAttentionProcessor:
        def __init__(self):
            pass

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Based on AttnProcessor2_0 logic

            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, kwargs.get('temb')) # Assuming temb might be in kwargs

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape

            # Attention mask preparation (copied from AttnProcessor2_0, but not passed to sageattn)
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # Keep the mask format expected by F.sdpa for reference, even if not used by sageattn here
                # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states, **kwargs)
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

            if attn.norm_cross:
                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, **kwargs)
            value = attn.to_v(encoder_hidden_states, **kwargs)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            # Reshape using view().transpose() like AttnProcessor2_0
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Apply Q/K norm if present
            if attn.norm_q is not None:
                 query = attn.norm_q(query)
            if attn.norm_k is not None:
                 key = attn.norm_k(key)

            # Call sageattn - NOTE: attention_mask is NOT passed
            hidden_states = sageattn(query, key, value, tensor_layout="HND", is_causal=False)

            # Reshape back using reshape().transpose() like AttnProcessor2_0
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            # Cast dtype
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, **kwargs)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    class SageFluxAttentionProcessor:
        def __init__(self):
            pass

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Similar refactoring as above, assuming Flux attention shares structure
            # Important: Check if Flux Attention has different norm attributes or steps

            residual = hidden_states
            # Flux might have specific norm attributes like norm_hid? Check attn obj
            # Add spatial_norm check if applicable
            # if attn.spatial_norm is not None: ...

            input_ndim = hidden_states.ndim
            if input_ndim == 4: # Assuming Flux might also receive 4D input sometimes
                 batch_size, channel, height, width = hidden_states.shape
                 hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape

            # Mask handling (prepared but not used by sageattn)
            if attention_mask is not None:
                 attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                 # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            # Group norm check if applicable
            # if attn.group_norm is not None: ...

            query = attn.to_q(hidden_states, **kwargs)
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

            # Cross norm check if applicable
            # if attn.norm_cross: ...

            key = attn.to_k(encoder_hidden_states, **kwargs)
            value = attn.to_v(encoder_hidden_states, **kwargs)

            # RoPE application likely happens INSIDE the attn block before processor

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            # Reshape using view().transpose()
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Apply Q/K norm if present in Flux Attention
            if hasattr(attn, 'norm_q') and attn.norm_q is not None:
                 query = attn.norm_q(query)
            if hasattr(attn, 'norm_k') and attn.norm_k is not None:
                 key = attn.norm_k(key)

            # Call sageattn - NOTE: attention_mask is NOT passed
            hidden_states = sageattn(query, key, value, tensor_layout="HND", is_causal=False)

            # Reshape back using reshape().transpose()
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            # Cast dtype
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, **kwargs)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # Residual connection check
            if hasattr(attn, 'residual_connection') and attn.residual_connection:
                hidden_states = hidden_states + residual

            # Rescale factor check
            if hasattr(attn, 'rescale_output_factor'):
                hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

else:
    class SageAttentionProcessor:
        def __call__(self, attn, hidden_states, **kwargs):
            raise RuntimeError("SageAttention is not available, cannot use SageAttentionProcessor.")

    class SageFluxAttentionProcessor:
        def __call__(self, attn, hidden_states, **kwargs):
            raise RuntimeError("SageAttention is not available, cannot use SageFluxAttentionProcessor.")
