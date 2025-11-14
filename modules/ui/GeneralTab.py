from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.TensorboardMode import TensorboardMode
from modules.util.ui import components

import customtkinter as ctk


# Sections
def _create_main_settings(train_ui_instance, parent):
    section = train_ui_instance._create_section(parent, "Main")

    main_settings = [
        (1, "Workspace Directory", "workspace_dir", "The directory where all files of this training run are saved"),
        (2, "Cache Directory", "cache_dir", "The directory where cached data is saved"),
    ]

    for row, label_text, var_name, tooltip in main_settings:
        components.label(section, row, 0, label_text, tooltip=tooltip, pad=(10, 5))
        components.path_entry(section, row, 1, train_ui_instance.ui_state, var_name, is_output=True,
                            path_type="directory", sticky="ew")


    switches_frame = ctk.CTkFrame(section, fg_color="transparent")
    switches_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 5))
    switches_frame.grid_columnconfigure(0, weight=0)
    switches_frame.grid_columnconfigure(2, weight=0)

    switch_configs = [
        (0, "continue_last_backup", "Continue last backup",
        "Automatically continues training from the last backup saved in <workspace>/backup"),
        (2, "only_cache", "Only Cache",
        "Only populate the cache, without any training"),
    ]

    for col, var_name, label, tooltip in switch_configs:
        components.labeled_switch(switches_frame, 0, col, train_ui_instance.ui_state, var_name,
                                label_text=label, tooltip=tooltip, layout="row")

    # Debugging
    components.label(section, 4, 0, "Debugging", font=("", 14, "bold"), pad=(10, 5))
    components.label(section, 5, 0, "Debug mode",
                    tooltip="Save debug information during the training into the debug directory", pad=(10, 5))
    components.switch(section, 5, 1, train_ui_instance.ui_state, "debug_mode")
    components.label(section, 6, 0, "Debug Directory",
                    tooltip="The directory where debug data is saved", pad=(10, 5))
    components.path_entry(section, 6, 1, train_ui_instance.ui_state, "debug_dir", is_output=True, path_type="directory")

    # Tensorboard
    components.label(section, 7, 0, "Tensorboard", font=("", 14, "bold"), pad=(10, 5))
    components.label(section, 8, 0, "Tensorboard Mode",
                    tooltip="Off: Disabled.\nTrain Only: Active only during training.\nAlways On: Always available.",
                    pad=(10, 5))
    components.options_kv(section, 8, 1, [
        ("Off", TensorboardMode.OFF),
        ("Train only", TensorboardMode.TRAIN_ONLY),
        ("Always on", TensorboardMode.ALWAYS_ON),
    ], train_ui_instance.ui_state, "tensorboard_mode", width=100, sticky="nw")

    components.label(section, 9, 0, "Tensorboard Port",
                    tooltip="Port to use for Tensorboard link", pad=(10, 5))
    components.entry(section, 9, 1, train_ui_instance.ui_state, "tensorboard_port", width=50, sticky="nw")

    components.label(section, 10, 0, "Expose Tensorboard",
                    tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)",
                    pad=(10, 5))
    components.switch(section, 10, 1, train_ui_instance.ui_state, "tensorboard_expose")

    section.columnconfigure(1, weight=1)

    return section

def _create_device_settings(train_ui_instance, parent):
    section = train_ui_instance._create_section(parent, "Devices Settings")

    device_settings = [
        (1, "Dataloader Threads", "dataloader_threads", 36,
        "Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching."),
        (2, "Train Device", "train_device", 66,
        'The device used for training. Can be "cuda", "cuda:0", "cuda:1" etc. Default:"cuda". Must be "cuda" for multi-GPU training.'),
        (3, "Temp Device", "temp_device", 66,
        'The device used to temporarily offload models while they are not used. Default:"cpu"'),
    ]

    for row, label_text, var_name, width, tooltip in device_settings:
        components.label(section, row, 0, label_text, tooltip=tooltip, pad=(10, 5))
        components.entry(section, row, 1, train_ui_instance.ui_state, var_name, width=width, sticky="nw")

    return section

def _create_validation_loss_section(train_ui_instance, parent):
    validation_section = train_ui_instance._create_section(parent, "Validation Loss")
    components.label(validation_section, 1, 0, "Enable",
                    tooltip="Enable validation loss and add new graph in tensorboard", pad=(10, 5))
    components.switch(validation_section, 1, 1, train_ui_instance.ui_state, "validation")
    components.label(validation_section, 2, 0, "Validate after",
                    tooltip="The interval used when validate training", pad=(10, 5))
    components.time_entry(validation_section, 2, 1, train_ui_instance.ui_state, "validate_after", "validate_after_unit",
                        width=60, unit_width=90, sticky="nw")
    return validation_section

def _create_device_vl_parent(train_ui_instance, parent):
    """Create a container with validation and device settings sections side by side"""
    settings_container = ctk.CTkFrame(parent, corner_radius=6, fg_color="transparent")
    settings_container.grid_columnconfigure(0, weight=0)
    settings_container.grid_columnconfigure(1, weight=0)

    # Create validation loss section on the left
    validation_section = _create_validation_loss_section(train_ui_instance, settings_container)
    validation_section.grid(row=0, column=0, sticky="nsew", padx=(0, 2.5), pady=0)

    # Create device settings section on the right
    device_section = _create_device_settings(train_ui_instance, settings_container)
    device_section.grid(row=0, column=1, sticky="nsew", padx=(2.5, 0), pady=0)

    return settings_container


def _create_multi_gpu_section(train_ui_instance, parent):
    """Extract multi-GPU section creation"""
    section = train_ui_instance._create_section(parent, "Multi-GPU")

    components.label(section, 1, 0, "Enable Multi-GPU",
                    tooltip="Enable multi-GPU training. Only intended for if you have multiple supported and identical devices.",
                    pad=(10, 5))
    components.switch(section, 1, 1, train_ui_instance.ui_state, "multi_gpu")

    # Store widgets that should be toggled
    widgets_to_toggle = []

    # Device Indexes
    label = components.label(section, 2, 0, "Device Indexes",
                    tooltip="Multi-GPU: A comma-separated list of device indexes. If empty, all your GPUs are used. With a list such as \"0,1,3,4\" you can omit a GPU, for example an on-board graphics GPU.",
                    pad=(10, 5))
    widgets_to_toggle.append(label)
    entry = components.entry(section, 2, 1, train_ui_instance.ui_state, "device_indexes", sticky="nw", width=250)
    widgets_to_toggle.append(entry)

    # Sequential model setup
    label = components.label(section, 3, 0, "Sequential model setup",
                    tooltip="Multi-GPU: If enabled, loading and setting up the model is done for each GPU one after the other. This is slower, but can reduce peak RAM usage.",
                    pad=(10, 5))
    widgets_to_toggle.append(label)
    switch = components.switch(section, 3, 1, train_ui_instance.ui_state, "sequential_model_setup")
    widgets_to_toggle.append(switch)

    # Gradient Reduce Precision
    label = components.label(section, 4, 0, "Gradient Reduce Precision",
                    tooltip="WEIGHT_DTYPE: Reduce gradients between GPUs in your weight data type; can be imprecise, but more efficient than float32\n"
                            "WEIGHT_DTYPE_STOCHASTIC: Sum up the gradients in your weight data type, but average them in float32 and stochastically round if your weight data type is bfloat16\n"
                            "FLOAT_32: Reduce gradients in float32\n"
                            "FLOAT_32_STOCHASTIC: Reduce gradients in float32; use stochastic rounding to bfloat16 if your weight data type is bfloat16",
                    wide_tooltip=True, pad=(10, 5))
    widgets_to_toggle.append(label)
    options = components.options(section, 4, 1, [str(x) for x in list(GradientReducePrecision)],
                    train_ui_instance.ui_state, "gradient_reduce_precision", width=250, sticky="nw")
    widgets_to_toggle.append(options)

    switches_frame = ctk.CTkFrame(section, fg_color="transparent")
    switches_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 5))
    switches_frame.grid_columnconfigure(0, weight=0)
    switches_frame.grid_columnconfigure(2, weight=0)
    widgets_to_toggle.append(switches_frame)

    switch_configs = [
        (0, "fused_gradient_reduce", "Fused Gradient Reduce",
        "Multi-GPU: Gradient synchronisation during the backward pass. Can be more efficient, especially with Async Gradient Reduce"),
        (2, "async_gradient_reduce", "Async Gradient Reduce",
        "Multi-GPU: Asynchroniously start the gradient reduce operations during the backward pass. Can be more efficient, but requires some VRAM."),
    ]

    for col, var_name, label, tooltip in switch_configs:
        components.labeled_switch(switches_frame, 0, col, train_ui_instance.ui_state, var_name,
                                label_text=label, tooltip=tooltip, layout="row")

    # Buffer size
    label = components.label(section, 6, 0, "Buffer size (MB)",
                    tooltip="Multi-GPU: Maximum VRAM for \"Async Gradient Reduce\", in megabytes. A multiple of this value can be needed if combined with \"Fused Back Pass\" and/or \"Layer offload fraction\"",
                    pad=(10, 5))
    widgets_to_toggle.append(label)
    entry = components.entry(section, 6, 1, train_ui_instance.ui_state, "async_gradient_reduce_buffer",
                    sticky="nw", width=70)
    widgets_to_toggle.append(entry)

    section.columnconfigure(1, weight=1)

    # Toggle visibility based on multi_gpu value
    def toggle_visibility():
        multi_gpu_enabled = train_ui_instance.ui_state.get_var("multi_gpu").get()
        for widget in widgets_to_toggle:
            if multi_gpu_enabled:
                widget.grid()
            else:
                widget.grid_remove()

    # Set up trace and initial state
    trace_id = train_ui_instance.ui_state.add_var_trace("multi_gpu", toggle_visibility)
    toggle_visibility()  # Set initial state

    # Store trace_id for cleanup
    if not hasattr(train_ui_instance, 'multi_gpu_trace_id'):
        train_ui_instance.multi_gpu_trace_id = trace_id

    return section




def _create_input_validation_section(train_ui_instance, parent):
    section = train_ui_instance._create_section(parent, "Input Validation")

    validation_settings = [
        (1, "Auto-correct Input", "validation_auto_correct",
        "Automatically correct file paths and extensions where possible"),
        (2, "Show Validation Tooltips", "validation_show_tooltips",
        "Show tooltips with validation messages for errors and warnings"),
        (3, "Use Friendly Names", "use_friendly_names",
        "Use friendly names instead of timestamps for auto-corrected/generated filenames"),
        (4, "Prevent Overwrite", "prevent_overwrite",
        "Automatically append a number or word to filenames that would overwrite existing files (requires Auto-correct Input)"),
        (5, "Auto-prefix", "auto_prefix",
        "Automatically prefix manually-entered filenames with the Save Filename Prefix (requires Auto-correct Input)"),
    ]

    for row, label_text, var_name, tooltip in validation_settings:
        components.label(section, row, 0, label_text, tooltip=tooltip, pad=(10, 5))
        components.switch(section, row, 1, train_ui_instance.ui_state, var_name)

    return section


# Layouts
def create_regular_layout(train_ui_instance):
    """Four-column layout with two fixed central columns to contain sections"""
    train_ui_instance._reset_layout()

    scrollable_frame = train_ui_instance.scrollable_frame
    scrollable_frame.columnconfigure(0, weight=1)
    scrollable_frame.columnconfigure(1, weight=0, minsize=520)
    scrollable_frame.columnconfigure(2, weight=0, minsize=320)
    scrollable_frame.columnconfigure(3, weight=1)

    scrollable_frame.rowconfigure(0, weight=0)
    scrollable_frame.rowconfigure(1, weight=0)
    scrollable_frame.rowconfigure(2, weight=0)
    scrollable_frame.rowconfigure(3, weight=0)

    main_section = _create_main_settings(train_ui_instance, scrollable_frame)
    main_section.grid(row=0, rowspan=4, column=1, sticky="nsew", padx=5, pady=5)

    # side by side mini sections
    combined_section = _create_device_vl_parent(train_ui_instance, scrollable_frame)
    combined_section.grid(row=0, column=2, sticky="nsew", padx=5, pady=0)

    multi_gpu_section = _create_multi_gpu_section(train_ui_instance, scrollable_frame)
    multi_gpu_section.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

    input_validation_section = _create_input_validation_section(train_ui_instance, scrollable_frame)
    input_validation_section.grid(row=3, column=2, sticky="nsew", padx=5, pady=5)


def create_small_layout(train_ui_instance):
    train_ui_instance._reset_layout()

    scrollable_frame = train_ui_instance.scrollable_frame
    scrollable_frame.columnconfigure(0, weight=1)

    main_section = _create_main_settings(train_ui_instance, scrollable_frame)
    main_section.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    validation_section = _create_validation_loss_section(train_ui_instance, scrollable_frame)
    validation_section.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    device_section = _create_device_settings(train_ui_instance, scrollable_frame)
    device_section.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

    multi_gpu_section = _create_multi_gpu_section(train_ui_instance, scrollable_frame)
    multi_gpu_section.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

    input_validation_section = _create_input_validation_section(train_ui_instance, scrollable_frame)
    input_validation_section.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
