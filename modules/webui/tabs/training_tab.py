"""Training settings tab for the Gradio WebUI.

Replicates the TrainingTab 3-column layout with all frames:
base (optimizer, LR), base2 (EMA, dtype, resolution),
UNet/Prior/Transformer/Text Encoder frames, noise, masked, loss, layer filter.
Popup windows are inline accordions.
"""

import gradio as gr

from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TimestepDistribution import TimestepDistribution


def create_training_tab():
    """Build the 'training' tab and return a dict of all components."""
    components = {}

    with gr.Row(equal_height=False):
        # ════════════════════════════════════════════════════════════
        # COLUMN 0: base + text encoder + embedding
        # ════════════════════════════════════════════════════════════
        with gr.Column():

            # ── Base Frame ──────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Training Base")

                components["optimizer.optimizer"] = gr.Dropdown(
                    label="Optimizer",
                    choices=[str(x) for x in list(Optimizer)],
                    value=str(Optimizer.ADAMW),
                    info="The type of optimizer",
                    interactive=True,
                )

                # Optimizer params accordion
                with gr.Accordion("Optimizer Settings", open=False) as opt_acc:
                    gr.Markdown(
                        "Optimizer-specific parameters are loaded dynamically "
                        "based on the selected optimizer. Edit the JSON config "
                        "directly for advanced parameter control."
                    )
                components["_optimizer_accordion"] = opt_acc

                components["learning_rate_scheduler"] = gr.Dropdown(
                    label="Learning Rate Scheduler",
                    choices=[str(x) for x in list(LearningRateScheduler)],
                    value=str(LearningRateScheduler.CONSTANT),
                    info="Learning rate scheduler that automatically changes the LR during training",
                    interactive=True,
                )

                # Scheduler params accordion
                with gr.Accordion("Scheduler Settings", open=False) as sched_acc:
                    components["custom_learning_rate_scheduler"] = gr.Textbox(
                        label="Class Name",
                        value="",
                        info="Python class path for the custom scheduler, e.g. module.ClassName",
                        interactive=True,
                    )
                    gr.Markdown(
                        "Additional scheduler key-value parameters can be "
                        "edited in the JSON config file directly."
                    )
                components["_scheduler_accordion"] = sched_acc

                components["learning_rate"] = gr.Number(
                    label="Learning Rate",
                    value=3e-4,
                    info="The base learning rate",
                    interactive=True,
                )
                components["learning_rate_warmup_steps"] = gr.Number(
                    label="Learning Rate Warmup Steps",
                    value=200,
                    precision=0,
                    info="Steps to gradually increase LR from 0. Values >1 = fixed steps, <=1 = percentage",
                    interactive=True,
                )
                components["learning_rate_min_factor"] = gr.Number(
                    label="Learning Rate Min Factor",
                    value=0.0,
                    info="For a factor of 0.1, the final LR will be 10% of the initial LR",
                    interactive=True,
                )
                components["learning_rate_cycles"] = gr.Number(
                    label="Learning Rate Cycles",
                    value=1.0,
                    info="Number of LR cycles (only if the scheduler supports cycles)",
                    interactive=True,
                )
                components["epochs"] = gr.Number(
                    label="Epochs",
                    value=100,
                    precision=0,
                    info="The number of epochs for a full training run",
                    interactive=True,
                )
                components["batch_size"] = gr.Number(
                    label="Local Batch Size",
                    value=1,
                    precision=0,
                    info="The batch size of one training step (per GPU)",
                    interactive=True,
                )
                components["gradient_accumulation_steps"] = gr.Number(
                    label="Accumulation Steps",
                    value=1,
                    precision=0,
                    info="Number of accumulation steps. Increase to trade batch size for speed",
                    interactive=True,
                )
                components["learning_rate_scaler"] = gr.Dropdown(
                    label="Learning Rate Scaler",
                    choices=[str(x) for x in list(LearningRateScaler)],
                    value=str(LearningRateScaler.NONE),
                    info="Type of LR scaling: LR * SQRT(selection)",
                    interactive=True,
                )
                components["clip_grad_norm"] = gr.Textbox(
                    label="Clip Grad Norm",
                    value="",
                    info="Clips the gradient norm. Leave empty to disable.",
                    interactive=True,
                )

            # ── Text Encoder Frame (single) ─────────────────────────
            with gr.Group(visible=True) as te_frame:
                gr.Markdown("#### Text Encoder")
                components["text_encoder.train"] = gr.Checkbox(
                    label="Train Text Encoder", value=True, interactive=True,
                )
                components["text_encoder.dropout_probability"] = gr.Number(
                    label="Caption Dropout Probability", value=0.0, interactive=True,
                )
                with gr.Row():
                    components["text_encoder.stop_training_after"] = gr.Number(
                        label="Stop Training After", value=0, precision=0, interactive=True,
                    )
                    components["text_encoder.stop_training_after_unit"] = gr.Dropdown(
                        label="Unit", choices=[str(x) for x in list(TimeUnit)],
                        value=str(TimeUnit.NEVER), interactive=True,
                    )
                components["text_encoder.learning_rate"] = gr.Textbox(
                    label="Text Encoder Learning Rate", value="",
                    info="Overrides the base LR. Leave empty for default.", interactive=True,
                )
                components["text_encoder_layer_skip"] = gr.Number(
                    label="Clip Skip", value=0, precision=0,
                    info="Additional clip layers to skip. 0 = model default", interactive=True,
                )
            components["_te_frame"] = te_frame

            # ── Text Encoder N Frames (1-4) ─────────────────────────
            for i in range(1, 5):
                suffix = "" if i == 1 else f"_{i}"
                attr = f"text_encoder{suffix}"
                with gr.Group(visible=False) as te_n_frame:
                    gr.Markdown(f"#### Text Encoder {i}")
                    components[f"{attr}.include"] = gr.Checkbox(
                        label=f"Include Text Encoder {i}", value=True, interactive=True,
                    )
                    components[f"{attr}.train"] = gr.Checkbox(
                        label=f"Train Text Encoder {i}", value=True, interactive=True,
                    )
                    components[f"{attr}.train_embedding"] = gr.Checkbox(
                        label=f"Train Text Encoder {i} Embedding", value=True, interactive=True,
                    )
                    components[f"{attr}.dropout_probability"] = gr.Number(
                        label="Dropout Probability", value=0.0, interactive=True,
                    )
                    with gr.Row():
                        components[f"{attr}.stop_training_after"] = gr.Number(
                            label="Stop Training After", value=0, precision=0, interactive=True,
                        )
                        components[f"{attr}.stop_training_after_unit"] = gr.Dropdown(
                            label="Unit", choices=[str(x) for x in list(TimeUnit)],
                            value=str(TimeUnit.NEVER), interactive=True,
                        )
                    components[f"{attr}.learning_rate"] = gr.Textbox(
                        label=f"Text Encoder {i} Learning Rate", value="", interactive=True,
                    )
                    components[f"text_encoder{suffix}_layer_skip"] = gr.Number(
                        label=f"Text Encoder {i} Clip Skip", value=0, precision=0, interactive=True,
                    )
                components[f"_te{i}_frame"] = te_n_frame

            # ── Embedding Frame ─────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Embeddings")
                components["embedding_learning_rate"] = gr.Textbox(
                    label="Embeddings Learning Rate", value="",
                    info="Overrides the base LR for embeddings", interactive=True,
                )
                components["preserve_embedding_norm"] = gr.Checkbox(
                    label="Preserve Embedding Norm", value=False,
                    info="Rescales each trained embedding to the median norm", interactive=True,
                )

        # ════════════════════════════════════════════════════════════
        # COLUMN 1: base2 + UNet/Prior/Transformer + noise
        # ════════════════════════════════════════════════════════════
        with gr.Column():

            # ── Base2 Frame ─────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Training Settings")
                components["ema"] = gr.Dropdown(
                    label="EMA",
                    choices=[str(x) for x in list(EMAMode)],
                    value=str(EMAMode.OFF),
                    info="EMA averages training progress over many steps",
                    interactive=True,
                )
                components["ema_decay"] = gr.Number(
                    label="EMA Decay", value=0.999,
                    info="Decay parameter of the EMA model", interactive=True,
                )
                components["ema_update_step_interval"] = gr.Number(
                    label="EMA Update Step Interval", value=5, precision=0, interactive=True,
                )

                components["gradient_checkpointing"] = gr.Dropdown(
                    label="Gradient Checkpointing",
                    choices=[str(x) for x in list(GradientCheckpointingMethod)],
                    value=str(GradientCheckpointingMethod.OFF),
                    info="Reduces memory usage but increases training time",
                    interactive=True,
                )

                # Offloading accordion
                with gr.Accordion("Offloading Settings", open=False):
                    components["enable_async_offloading"] = gr.Checkbox(
                        label="Async Offloading", value=False, interactive=True,
                    )
                    components["enable_activation_offloading"] = gr.Checkbox(
                        label="Offload Activations", value=False, interactive=True,
                    )
                    components["layer_offload_fraction"] = gr.Number(
                        label="Layer Offload Fraction", value=0.0,
                        info="0 = disabled, values between 0 and 1", interactive=True,
                    )

                components["train_dtype"] = gr.Dropdown(
                    label="Train Data Type",
                    choices=[
                        ("float32", str(DataType.FLOAT_32)),
                        ("float16", str(DataType.FLOAT_16)),
                        ("bfloat16", str(DataType.BFLOAT_16)),
                        ("tfloat32", str(DataType.TFLOAT_32)),
                    ],
                    value=str(DataType.FLOAT_32),
                    info="Mixed precision data type used for training",
                    interactive=True,
                )
                components["fallback_train_dtype"] = gr.Dropdown(
                    label="Fallback Train Data Type",
                    choices=[
                        ("float32", str(DataType.FLOAT_32)),
                        ("bfloat16", str(DataType.BFLOAT_16)),
                    ],
                    value=str(DataType.FLOAT_32),
                    interactive=True,
                )
                components["enable_autocast_cache"] = gr.Checkbox(
                    label="Autocast Cache", value=True,
                    info="Disabling reduces memory but increases training time", interactive=True,
                )
                components["resolution"] = gr.Textbox(
                    label="Resolution", value="512",
                    info="Training resolution. Comma-separated for multiple, or WxH for exact",
                    interactive=True,
                )
                components["force_circular_padding"] = gr.Checkbox(
                    label="Force Circular Padding", value=False,
                    info="Enables circular padding for seamless images", interactive=True,
                )

            # ── UNet Frame ──────────────────────────────────────────
            with gr.Group(visible=True) as unet_frame:
                gr.Markdown("#### UNet")
                components["unet.train"] = gr.Checkbox(
                    label="Train UNet", value=True, interactive=True,
                )
                with gr.Row():
                    components["unet.stop_training_after"] = gr.Number(
                        label="Stop Training After", value=0, precision=0, interactive=True,
                    )
                    components["unet.stop_training_after_unit"] = gr.Dropdown(
                        label="Unit", choices=[str(x) for x in list(TimeUnit)],
                        value=str(TimeUnit.NEVER), interactive=True,
                    )
                components["unet.learning_rate"] = gr.Textbox(
                    label="UNet Learning Rate", value="", interactive=True,
                )
                components["rescale_noise_scheduler_to_zero_terminal_snr"] = gr.Checkbox(
                    label="Rescale Noise Scheduler + V-pred", value=False, interactive=True,
                )
            components["_unet_frame"] = unet_frame

            # ── Prior Frame ─────────────────────────────────────────
            with gr.Group(visible=False) as prior_frame:
                gr.Markdown("#### Prior")
                components["prior.train"] = gr.Checkbox(
                    label="Train Prior", value=True, interactive=True,
                )
                with gr.Row():
                    components["prior.stop_training_after"] = gr.Number(
                        label="Stop Training After", value=0, precision=0, interactive=True,
                    )
                    components["prior.stop_training_after_unit"] = gr.Dropdown(
                        label="Unit", choices=[str(x) for x in list(TimeUnit)],
                        value=str(TimeUnit.NEVER), interactive=True,
                    )
                components["prior.learning_rate"] = gr.Textbox(
                    label="Prior Learning Rate", value="", interactive=True,
                )
            components["_prior_frame"] = prior_frame

            # ── Transformer Frame ───────────────────────────────────
            with gr.Group(visible=False) as transformer_frame:
                gr.Markdown("#### Transformer")
                components["transformer.train"] = gr.Checkbox(
                    label="Train Transformer", value=True, interactive=True,
                )
                with gr.Row():
                    components["transformer.stop_training_after"] = gr.Number(
                        label="Stop Training After", value=0, precision=0, interactive=True,
                    )
                    components["transformer.stop_training_after_unit"] = gr.Dropdown(
                        label="Unit", choices=[str(x) for x in list(TimeUnit)],
                        value=str(TimeUnit.NEVER), interactive=True,
                    )
                components["transformer.learning_rate"] = gr.Textbox(
                    label="Transformer Learning Rate", value="", interactive=True,
                )
                components["transformer.attention_mask"] = gr.Checkbox(
                    label="Force Attention Mask", value=False, interactive=True,
                )
                components["transformer.guidance_scale"] = gr.Number(
                    label="Guidance Scale", value=1.0, interactive=True,
                )
            components["_transformer_frame"] = transformer_frame

            # ── Noise Frame ─────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Noise")
                components["offset_noise_weight"] = gr.Number(
                    label="Offset Noise Weight", value=0.0, interactive=True,
                )
                components["generalized_offset_noise"] = gr.Checkbox(
                    label="Generalized Offset Noise", value=False, interactive=True,
                )
                components["perturbation_noise_weight"] = gr.Number(
                    label="Perturbation Noise Weight", value=0.0, interactive=True,
                )
                components["timestep_distribution"] = gr.Dropdown(
                    label="Timestep Distribution",
                    choices=[str(x) for x in list(TimestepDistribution)],
                    value=str(TimestepDistribution.UNIFORM),
                    info="Function to sample timesteps during training",
                    interactive=True,
                )

                # Timestep distribution accordion
                with gr.Accordion("Timestep Distribution Settings", open=False):
                    components["min_noising_strength"] = gr.Number(
                        label="Min Noising Strength", value=0.0,
                        info="Minimum noising strength during training", interactive=True,
                    )
                    components["max_noising_strength"] = gr.Number(
                        label="Max Noising Strength", value=1.0,
                        info="Maximum noising strength during training", interactive=True,
                    )
                    components["noising_weight"] = gr.Number(
                        label="Noising Weight", value=1.0, interactive=True,
                    )
                    components["noising_bias"] = gr.Number(
                        label="Noising Bias", value=0.0, interactive=True,
                    )
                    components["timestep_shift"] = gr.Number(
                        label="Timestep Shift", value=1.0, interactive=True,
                    )
                    components["dynamic_timestep_shifting"] = gr.Checkbox(
                        label="Dynamic Timestep Shifting", value=False,
                        info="Dynamically shift the timestep distribution based on resolution",
                        interactive=True,
                    )

        # ════════════════════════════════════════════════════════════
        # COLUMN 2: masked + loss + layer filter
        # ════════════════════════════════════════════════════════════
        with gr.Column():

            # ── Masked Training Frame ───────────────────────────────
            with gr.Group():
                gr.Markdown("#### Masked Training")
                components["masked_training"] = gr.Checkbox(
                    label="Masked Training", value=False, interactive=True,
                )
                components["unmasked_probability"] = gr.Number(
                    label="Unmasked Probability", value=0.0, interactive=True,
                )
                components["unmasked_weight"] = gr.Number(
                    label="Unmasked Weight", value=0.01, interactive=True,
                )
                components["normalize_masked_area_loss"] = gr.Checkbox(
                    label="Normalize Masked Area Loss", value=False, interactive=True,
                )
                components["masked_prior_preservation_weight"] = gr.Number(
                    label="Masked Prior Preservation Weight", value=0.0, interactive=True,
                )
                components["custom_conditioning_image"] = gr.Checkbox(
                    label="Custom Conditioning Image", value=False, interactive=True,
                )

            # ── Loss Frame ──────────────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Loss")
                with gr.Row():
                    components["mse_strength"] = gr.Number(
                        label="MSE Strength", value=1.0, interactive=True,
                    )
                    components["mae_strength"] = gr.Number(
                        label="MAE Strength", value=0.0, interactive=True,
                    )
                with gr.Row():
                    components["log_cosh_strength"] = gr.Number(
                        label="log-cosh Strength", value=0.0, interactive=True,
                    )
                    components["huber_strength"] = gr.Number(
                        label="Huber Strength", value=0.0, interactive=True,
                    )
                components["huber_delta"] = gr.Number(
                    label="Huber Delta", value=0.1, interactive=True,
                )
                components["vb_loss_strength"] = gr.Number(
                    label="VB Strength", value=1.0,
                    info="Only used for PixArt models", interactive=True,
                )
                components["loss_weight_fn"] = gr.Dropdown(
                    label="Loss Weight Function",
                    choices=[str(x) for x in list(LossWeight)],
                    value=str(LossWeight.CONSTANT),
                    interactive=True,
                )
                components["loss_weight_strength"] = gr.Number(
                    label="Gamma", value=5.0, interactive=True,
                )
                components["loss_scaler"] = gr.Dropdown(
                    label="Loss Scaler",
                    choices=[str(x) for x in list(LossScaler)],
                    value=str(LossScaler.NONE),
                    interactive=True,
                )

            # ── Layer Filter Frame ──────────────────────────────────
            with gr.Group():
                gr.Markdown("#### Layer Filter")
                with gr.Row():
                    components["layer_filter_preset"] = gr.Dropdown(
                        label="Layer Filter Preset",
                        choices=["full"],
                        value="full",
                        interactive=True,
                    )
                    components["layer_filter"] = gr.Textbox(
                        label="Layer Filter",
                        value="",
                        info="Comma-separated layer names",
                        interactive=True,
                    )
                    components["layer_filter_regex"] = gr.Checkbox(
                        label="Regex", value=False, interactive=True,
                    )

    return components
