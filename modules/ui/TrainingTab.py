from modules.modelSetup.BaseChromaSetup import PRESETS as chroma_presets
from modules.modelSetup.BaseFluxSetup import PRESETS as flux_presets
from modules.modelSetup.BaseHiDreamSetup import PRESETS as hidream_presets
from modules.modelSetup.BaseHunyuanVideoSetup import PRESETS as hunyuan_video_presets
from modules.modelSetup.BasePixArtAlphaSetup import PRESETS as pixart_presets
from modules.modelSetup.BaseQwenSetup import PRESETS as qwen_presets
from modules.modelSetup.BaseSanaSetup import PRESETS as sana_presets
from modules.modelSetup.BaseStableDiffusion3Setup import PRESETS as sd3_presets
from modules.modelSetup.BaseStableDiffusionSetup import PRESETS as sd_presets
from modules.modelSetup.BaseStableDiffusionXLSetup import PRESETS as sdxl_presets
from modules.modelSetup.BaseWuerstchenSetup import PRESETS as sc_presets
from modules.ui.OffloadingWindow import OffloadingWindow
from modules.ui.OptimizerParamsWindow import OptimizerParamsWindow
from modules.ui.SchedulerParamsWindow import SchedulerParamsWindow
from modules.ui.TimestepDistributionWindow import TimestepDistributionWindow
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.optimizer_util import change_optimizer
from modules.util.ui import components
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class TrainingTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.scroll_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_frame.grid_columnconfigure(0, weight=1)
        self.scroll_frame.grid_columnconfigure(1, weight=1)
        self.scroll_frame.grid_columnconfigure(2, weight=1)

        column_0 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_0.grid(row=0, column=0, sticky="nsew")
        column_0.grid_columnconfigure(0, weight=1)

        column_1 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_1.grid(row=0, column=1, sticky="nsew")
        column_1.grid_columnconfigure(0, weight=1)

        column_2 = ctk.CTkFrame(master=self.scroll_frame, corner_radius=0, fg_color="transparent")
        column_2.grid(row=0, column=2, sticky="nsew")
        column_2.grid_columnconfigure(0, weight=1)

        if self.train_config.model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui(column_0, column_1, column_2)
        if self.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_flux():
            self.__setup_flux_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_chroma():
            self.__setup_chroma_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_qwen():
            self.__setup_qwen_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_sana():
            self.__setup_sana_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_hunyuan_video():
            self.__setup_hunyuan_video_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_hi_dream():
            self.__setup_hi_dream_ui(column_0, column_1, column_2)


    def __setup_stable_diffusion_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_unet_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2, supports_generalized_offset_noise=True)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_stable_diffusion_3_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_n_frame(column_0, 1, i=1, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 2, i=2, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 3, i=3, supports_include=True)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_stable_diffusion_xl_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_n_frame(column_0, 1, i=1)
        self.__create_text_encoder_n_frame(column_0, 2, i=2)
        self.__create_embedding_frame(column_0, 3)

        self.__create_base2_frame(column_1, 0)
        self.__create_unet_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2, supports_generalized_offset_noise=True)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_wuerstchen_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_prior_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 0)
        self.__create_loss_frame(column_2, 1)
        self.__create_layer_frame(column_2, 2)

    def __setup_pixart_alpha_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2, supports_vb_loss=True)
        self.__create_layer_frame(column_2, 3)

    def __setup_flux_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_n_frame(column_0, 1, i=1, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 2, i=2, supports_include=True, supports_sequence_length=True)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1, supports_guidance_scale=True)
        self.__create_noise_frame(column_1, 2, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_chroma_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_qwen_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1, supports_clip_skip=False)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1, supports_guidance_scale=False, supports_force_attention_mask=False)
        self.__create_noise_frame(column_1, 2, supports_dynamic_timestep_shifting=True)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_sana_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)
        self.__create_embedding_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_transformer_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_hunyuan_video_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_n_frame(column_0, 1, i=1, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 2, i=2, supports_include=True)
        self.__create_embedding_frame(column_0, 4)

        self.__create_base2_frame(column_1, 0, video_training_enabled=True)
        self.__create_transformer_frame(column_1, 1, supports_guidance_scale=True)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __setup_hi_dream_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_n_frame(column_0, 1, i=1, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 2, i=2, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 3, i=3, supports_include=True)
        self.__create_text_encoder_n_frame(column_0, 4, i=4, supports_include=True, supports_layer_skip=False)
        self.__create_embedding_frame(column_0, 5)

        self.__create_base2_frame(column_1, 0, video_training_enabled=True)
        self.__create_transformer_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2)
        self.__create_layer_frame(column_2, 3)

    def __create_base_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # optimizer
        components.label(frame, 0, 0, "Optimizer",
                         tooltip="The type of optimizer")
        components.options_adv(frame, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer.optimizer",
                               command=self.__restore_optimizer_config, adv_command=self.__open_optimizer_params_window)

        # learning rate scheduler
        # Wackiness will ensue when reloading configs if we don't check and clear this first.
        if hasattr(self, "lr_scheduler_comp"):
            delattr(self, "lr_scheduler_comp")
            delattr(self, "lr_scheduler_adv_comp")
        components.label(frame, 1, 0, "Learning Rate Scheduler",
                         tooltip="Learning rate scheduler that automatically changes the learning rate during training")
        _, d = components.options_adv(frame, 1, 1, [str(x) for x in list(LearningRateScheduler)], self.ui_state,
                                      "learning_rate_scheduler", command=self.__restore_scheduler_config,
                                      adv_command=self.__open_scheduler_params_window)
        self.lr_scheduler_comp = d['component']
        self.lr_scheduler_adv_comp = d['button_component']
        # Initial call requires the presence of self.lr_scheduler_adv_comp.
        self.__restore_scheduler_config(self.ui_state.get_var("learning_rate_scheduler").get())

        # learning rate
        components.label(frame, 2, 0, "Learning Rate",
                         tooltip="The base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "learning_rate")

        # learning rate warmup steps
        components.label(frame, 3, 0, "Learning Rate Warmup Steps",
                         tooltip="The number of steps it takes to gradually increase the learning rate from 0 to the specified learning rate. Values >1 are interpeted as a fixed number of steps, values <=1 are intepreted as a percentage of the total training steps (ex. 0.2 = 20% of the total step count)")
        components.entry(frame, 3, 1, self.ui_state, "learning_rate_warmup_steps")

        # learning rate min factor
        components.label(frame, 4, 0, "Learning Rate Min Factor",
                         tooltip="Unit = float. Method = percentage. For a factor of 0.1, the final LR will be 10% of the initial LR. If the initial LR is 1e-4, the final LR will be 1e-5.")
        components.entry(frame, 4, 1, self.ui_state, "learning_rate_min_factor")

        # learning rate cycles
        components.label(frame, 5, 0, "Learning Rate Cycles",
                         tooltip="The number of learning rate cycles. This is only applicable if the learning rate scheduler supports cycles")
        components.entry(frame, 5, 1, self.ui_state, "learning_rate_cycles")

        # epochs
        components.label(frame, 6, 0, "Epochs",
                         tooltip="The number of epochs for a full training run")
        components.entry(frame, 6, 1, self.ui_state, "epochs")

        # batch size
        components.label(frame, 7, 0, "Local Batch Size",
                         tooltip="The batch size of one training step. If you use multiple GPUs, this is the batch size of each GPU (local batch size).")
        components.entry(frame, 7, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(frame, 8, 0, "Accumulation Steps",
                         tooltip="Number of accumulation steps. Increase this number to trade batch size for training speed")
        components.entry(frame, 8, 1, self.ui_state, "gradient_accumulation_steps")

        # Learning Rate Scaler
        components.label(frame, 9, 0, "Learning Rate Scaler",
                         tooltip="Selects the type of learning rate scaling to use during training. Functionally equated as: LR * SQRT(selection)")
        components.options(frame, 9, 1, [str(x) for x in list(LearningRateScaler)], self.ui_state,
                           "learning_rate_scaler")

        # clip grad norm
        components.label(frame, 10, 0, "Clip Grad Norm",
                         tooltip="Clips the gradient norm. Leave empty to disable gradient clipping.")
        components.entry(frame, 10, 1, self.ui_state, "clip_grad_norm")

    def __create_base2_frame(self, master, row, video_training_enabled: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        row = 0

        # ema
        components.label(frame, row, 0, "EMA",
                         tooltip="EMA averages the training progress over many steps, better preserving different concepts in big datasets")
        components.options(frame, row, 1, [str(x) for x in list(EMAMode)], self.ui_state, "ema")
        row += 1

        # ema decay
        components.label(frame, row, 0, "EMA Decay",
                         tooltip="Decay parameter of the EMA model. Higher numbers will average more steps. For datasets of hundreds or thousands of images, set this to 0.9999. For smaller datasets, set it to 0.999 or even 0.998")
        components.entry(frame, row, 1, self.ui_state, "ema_decay")
        row += 1

        # ema update step interval
        components.label(frame, row, 0, "EMA Update Step Interval",
                         tooltip="Number of steps between EMA update steps")
        components.entry(frame, row, 1, self.ui_state, "ema_update_step_interval")
        row += 1

        # gradient checkpointing
        components.label(frame, row, 0, "Gradient checkpointing",
                         tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        components.options_adv(frame, row, 1, [str(x) for x in list(GradientCheckpointingMethod)], self.ui_state,
                           "gradient_checkpointing", adv_command=self.__open_offloading_window)
        row += 1

        # gradient checkpointing layer offloading
        components.label(frame, row, 0, "Layer offload fraction",
                         tooltip="Enables offloading of individual layers during training to reduce VRAM usage. Increases training time and uses more RAM. Only available if checkpointing is set to CPU_OFFLOADED. values between 0 and 1, 0=disabled")
        components.entry(frame, row, 1, self.ui_state, "layer_offload_fraction")
        row += 1

        # train dtype
        components.label(frame, row, 0, "Train Data Type",
                         tooltip="The mixed precision data type used for training. This can increase training speed, but reduces precision")
        components.options_kv(frame, row, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], self.ui_state, "train_dtype")
        row += 1

        # fallback train dtype
        components.label(frame, row, 0, "Fallback Train Data Type",
                         tooltip="The mixed precision data type used for training stages that don't support float16 data types. This can increase training speed, but reduces precision")
        components.options_kv(frame, row, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "fallback_train_dtype")
        row += 1

        # autocast cache
        components.label(frame, row, 0, "Autocast Cache",
                         tooltip="Enables the autocast cache. Disabling this reduces memory usage, but increases training time")
        components.switch(frame, row, 1, self.ui_state, "enable_autocast_cache")
        row += 1

        # resolution
        components.label(frame, row, 0, "Resolution",
                         tooltip="The resolution used for training. Optionally specify multiple resolutions separated by a comma, or a single exact resolution in the format <width>x<height>")
        components.entry(frame, row, 1, self.ui_state, "resolution")
        row += 1

        # frames
        if video_training_enabled:
            components.label(frame, row, 0, "Frames",
                             tooltip="The number of frames used for training.")
            components.entry(frame, row, 1, self.ui_state, "frames")
            row += 1

        # force circular padding
        components.label(frame, row, 0, "Force Circular Padding",
                         tooltip="Enables circular padding for all conv layers to better train seamless images")
        components.switch(frame, row, 1, self.ui_state, "force_circular_padding")

    def __create_text_encoder_frame(self, master, row, supports_clip_skip=True):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train text encoder
        components.label(frame, 0, 0, "Train Text Encoder",
                         tooltip="Enables training the text encoder model")
        components.switch(frame, 0, 1, self.ui_state, "text_encoder.train")

        # dropout
        components.label(frame, 1, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder conditioning")
        components.entry(frame, 1, 1, self.ui_state, "text_encoder.dropout_probability")

        # train text encoder epochs
        components.label(frame, 2, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder")
        components.time_entry(frame, 2, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)

        # text encoder learning rate
        components.label(frame, 3, 0, "Text Encoder Learning Rate",
                         tooltip="The learning rate of the text encoder. Overrides the base learning rate")
        components.entry(frame, 3, 1, self.ui_state, "text_encoder.learning_rate")

        if supports_clip_skip:
            # text encoder layer skip (clip skip)
            components.label(frame, 4, 0, "Clip Skip",
                             tooltip="The number of additional clip layers to skip. 0 = the model default")
            components.entry(frame, 4, 1, self.ui_state, "text_encoder_layer_skip")

    def __create_text_encoder_n_frame(
            self,
            master,
            row: int,
            i: int,
            supports_include: bool = False,
            supports_layer_skip: bool = True,
            supports_sequence_length: bool = False,
    ):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        row = 0

        suffix = f"_{i}" if i > 1 else ""

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, f"Include Text Encoder {i}",
                             tooltip=f"Includes text encoder {i} in the training run")
            components.switch(frame, row, 1, self.ui_state, f"text_encoder{suffix}.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, f"Train Text Encoder {i}",
                         tooltip=f"Enables training the text encoder {i} model")
        components.switch(frame, row, 1, self.ui_state, f"text_encoder{suffix}.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, f"Train Text Encoder {i} Embedding",
                         tooltip=f"Enables training embeddings for the text encoder {i} model")
        components.switch(frame, row, 1, self.ui_state, f"text_encoder{suffix}.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "Dropout Probability",
                         tooltip=f"The Probability for dropping the text encoder {i} conditioning")
        components.entry(frame, row, 1, self.ui_state, f"text_encoder{suffix}.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "Stop Training After",
                         tooltip=f"When to stop training the text encoder {i}")
        components.time_entry(frame, row, 1, self.ui_state, f"text_encoder{suffix}.stop_training_after",
                              f"text_encoder{suffix}.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, f"Text Encoder {i} Learning Rate",
                         tooltip=f"The learning rate of the text encoder {i}. Overrides the base learning rate")
        components.entry(frame, row, 1, self.ui_state, f"text_encoder{suffix}.learning_rate")
        row += 1

        if supports_layer_skip:
            # text encoder layer skip (clip skip)
            components.label(frame, row, 0, f"Text Encoder {i} Clip Skip",
                             tooltip="The number of additional clip layers to skip. 0 = the model default")
            components.entry(frame, row, 1, self.ui_state, f"text_encoder{suffix}_layer_skip")
            row += 1

        if supports_sequence_length:
            # text encoder sequence length
            components.label(frame, row, 0, f"Text Encoder {i} Sequence Length",
                             tooltip="Overrides the number of tokens used for captions. If empty, the model default is used, which is 512 on Flux. Comfy samples with 256 tokens though. 77 is the default only for backwards compatibility.")
            components.entry(frame, row, 1, self.ui_state, f"text_encoder{suffix}_sequence_length")
            row += 1

    def __create_embedding_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")

        # embedding learning rate
        components.label(frame, 0, 0, "Embeddings Learning Rate",
                         tooltip="The learning rate of embeddings. Overrides the base learning rate")
        components.entry(frame, 0, 1, self.ui_state, "embedding_learning_rate")

        # preserve embedding norm
        components.label(frame, 1, 0, "Preserve Embedding Norm",
                         tooltip="Rescales each trained embedding to the median embedding norm")
        components.switch(frame, 1, 1, self.ui_state, "preserve_embedding_norm")

    def __create_unet_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train unet
        components.label(frame, 0, 0, "Train UNet",
                         tooltip="Enables training the UNet model")
        components.switch(frame, 0, 1, self.ui_state, "unet.train")

        # train unet epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the UNet")
        components.time_entry(frame, 1, 1, self.ui_state, "unet.stop_training_after", "unet.stop_training_after_unit",
                              supports_time_units=False)

        # unet learning rate
        components.label(frame, 2, 0, "UNet Learning Rate",
                         tooltip="The learning rate of the UNet. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "unet.learning_rate")

        # rescale noise scheduler to zero terminal SNR
        rescale_label = components.label(frame, 3, 0, "Rescale Noise Scheduler + V-pred",
                         tooltip="Rescales the noise scheduler to a zero terminal signal to noise ratio and switches the model to a v-prediction target")
        rescale_label.configure(wraplength=130, justify="left")
        components.switch(frame, 3, 1, self.ui_state, "rescale_noise_scheduler_to_zero_terminal_snr")

    def __create_prior_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train prior
        components.label(frame, 0, 0, "Train Prior",
                         tooltip="Enables training the Prior model")
        components.switch(frame, 0, 1, self.ui_state, "prior.train")

        # train prior epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the Prior")
        components.time_entry(frame, 1, 1, self.ui_state, "prior.stop_training_after", "prior.stop_training_after_unit",
                              supports_time_units=False)

        # prior learning rate
        components.label(frame, 2, 0, "Prior Learning Rate",
                         tooltip="The learning rate of the Prior. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "prior.learning_rate")

    def __create_transformer_frame(self, master, row, supports_guidance_scale: bool = False, supports_force_attention_mask: bool = True):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train transformer
        components.label(frame, 0, 0, "Train Transformer",
                         tooltip="Enables training the Transformer model")
        components.switch(frame, 0, 1, self.ui_state, "transformer.train")

        # train transformer epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the Transformer")
        components.time_entry(frame, 1, 1, self.ui_state, "transformer.stop_training_after", "transformer.stop_training_after_unit",
                              supports_time_units=False)

        # transformer learning rate
        components.label(frame, 2, 0, "Transformer Learning Rate",
                         tooltip="The learning rate of the Transformer. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "transformer.learning_rate")

        if supports_force_attention_mask:
            # transformer learning rate
            components.label(frame, 3, 0, "Force Attention Mask",
                             tooltip="Force enables passing of a text embedding attention mask to the transformer. This can improve training on shorter captions.")
            components.switch(frame, 3, 1, self.ui_state, "transformer.attention_mask")

        if supports_guidance_scale:
            # guidance scale
            components.label(frame, 4, 0, "Guidance Scale",
                             tooltip="The guidance scale of guidance distilled models passed to the transformer during training.")
            components.entry(frame, 4, 1, self.ui_state, "transformer.guidance_scale")

    def __create_noise_frame(self, master, row, supports_generalized_offset_noise: bool = False, supports_dynamic_timestep_shifting: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # offset noise weight
        components.label(frame, 0, 0, "Offset Noise Weight",
                         tooltip="The weight of offset noise added to each training step")
        components.entry(frame, 0, 1, self.ui_state, "offset_noise_weight")

        if supports_generalized_offset_noise:
            # generalized offset noise weight
            generalised_offset_label = components.label(frame, 1, 0, "Generalized Offset Noise",
                            tooltip="Per-timestep 'brightness knob' instead of a fixed offset - steadier training, better starts, and improved very dark/bright images. Compatible with V-pred and Eps-pred. Start with 0.02 and adjust as needed.")
            generalised_offset_label.configure(wraplength=130, justify="left")
            components.switch(frame, 1, 1, self.ui_state, "generalized_offset_noise")

        # perturbation noise weight
        components.label(frame, 2, 0, "Perturbation Noise Weight",
                         tooltip="The weight of perturbation noise added to each training step")
        components.entry(frame, 2, 1, self.ui_state, "perturbation_noise_weight")

        # timestep distribution
        components.label(frame, 3, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        components.options_adv(frame, 3, 1, [str(x) for x in list(TimestepDistribution)], self.ui_state, "timestep_distribution",
                               adv_command=self.__open_timestep_distribution_window)

        # min noising strength
        components.label(frame, 4, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        components.entry(frame, 4, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(frame, 5, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        components.entry(frame, 5, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(frame, 6, 0, "Noising Weight",
                         tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(frame, 6, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(frame, 7, 0, "Noising Bias",
                         tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(frame, 7, 1, self.ui_state, "noising_bias")

        # timestep shift
        components.label(frame, 8, 0, "Timestep Shift",
                         tooltip="Shift the timestep distribution. Use the preview to see more details.")
        components.entry(frame, 8, 1, self.ui_state, "timestep_shift")

        if supports_dynamic_timestep_shifting:
            # dynamic timestep shifting
            components.label(frame, 9, 0, "Dynamic Timestep Shifting",
                             tooltip="Dynamically shift the timestep distribution based on resolution.")
            components.switch(frame, 9, 1, self.ui_state, "dynamic_timestep_shifting")



    def __create_masked_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # Masked Training
        components.label(frame, 0, 0, "Masked Training",
                         tooltip="Masks the training samples to let the model focus on certain parts of the image. When enabled, one mask image is loaded for each training sample.")
        components.switch(frame, 0, 1, self.ui_state, "masked_training")

        # unmasked probability
        components.label(frame, 1, 0, "Unmasked Probability",
                         tooltip="When masked training is enabled, specifies the number of training steps done on unmasked samples")
        components.entry(frame, 1, 1, self.ui_state, "unmasked_probability")

        # unmasked weight
        components.label(frame, 2, 0, "Unmasked Weight",
                         tooltip="When masked training is enabled, specifies the loss weight of areas outside the masked region")
        components.entry(frame, 2, 1, self.ui_state, "unmasked_weight")

        # normalize masked area loss
        components.label(frame, 3, 0, "Normalize Masked Area Loss",
                         tooltip="When masked training is enabled, normalizes the loss for each sample based on the sizes of the masked region")
        components.switch(frame, 3, 1, self.ui_state, "normalize_masked_area_loss")

        # masked prior preservation
        components.label(frame, 4, 0, "Masked Prior Preservation Weight",
                         tooltip="Preserves regions outside the mask using the original untrained model output as a target. Only available for LoRA training. If enabled, use a low unmasked weight.")
        components.entry(frame, 4, 1, self.ui_state, "masked_prior_preservation_weight")

        # use custom conditioning image
        components.label(frame, 5, 0, "Custom Conditioning Image",
                         tooltip="When custom conditioning image is enabled, will use png postfix with -condlabel instead of automatically generated.It's suitable for special scenarios, such as object removal, allowing the model to learn a certain behavior concept")
        components.switch(frame, 5, 1, self.ui_state, "custom_conditioning_image")

    def __create_loss_frame(self, master, row, supports_vb_loss: bool = False):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # MSE Strength
        components.label(frame, 0, 0, "MSE Strength",
                         tooltip="Mean Squared Error strength for custom loss settings. Strengths should generally sum to 1.")
        components.entry(frame, 0, 1, self.ui_state, "mse_strength")

        # MAE Strength
        components.label(frame, 1, 0, "MAE Strength",
                         tooltip="Mean Absolute Error strength for custom loss settings. Strengths should generally sum to 1.")
        components.entry(frame, 1, 1, self.ui_state, "mae_strength")

        # log-cosh Strength
        components.label(frame, 2, 0, "log-cosh Strength",
                         tooltip="Log - Hyperbolic cosine Error strength for custom loss settings. Strengths should generally sum to 1.")
        components.entry(frame, 2, 1, self.ui_state, "log_cosh_strength")

        # Huber Strength
        components.label(frame, 3, 0, "Huber Strength",
                         tooltip="Huber loss strength for custom loss settings. Less sensitive to outliers than MSE. Strengths should generally sum to 1.")
        components.entry(frame, 3, 1, self.ui_state, "huber_strength")

        # Huber Delta
        components.label(frame, 4, 0, "Huber Delta",
                         tooltip="Delta parameter for huber loss")
        components.entry(frame, 4, 1, self.ui_state, "huber_delta")

        if supports_vb_loss:
            # VB Strength
            components.label(frame, 5, 0, "VB Strength",
                             tooltip="Variational lower-bound strength for custom loss settings. Should be set to 1 for variational diffusion models")
            components.entry(frame, 5, 1, self.ui_state, "vb_loss_strength")

        # Loss Weight function
        components.label(frame, 6, 0, "Loss Weight Function",
                         tooltip="Choice of loss weight function. Can help the model learn details more accurately.")
        components.options(frame, 6, 1, [str(x) for x in list(LossWeight)
                                         if x.supports_flow_matching() == self.train_config.model_type.is_flow_matching()
                                            or x == LossWeight.CONSTANT
                                        ],
                                        self.ui_state, "loss_weight_fn")

        row = 7

        # Loss weight strength
        if not self.train_config.model_type.is_flow_matching():
            components.label(frame, row, 0, "Gamma",
                             tooltip="Inverse strength of loss weighting. Range: 1-20, only applies to Min SNR and P2.")
            components.entry(frame, row, 1, self.ui_state, "loss_weight_strength")
            row += 1

        # Loss Scaler
        components.label(frame, row, 0, "Loss Scaler",
                         tooltip="Selects the type of loss scaling to use during training. Functionally equated as: Loss * selection")
        components.options(frame, row, 1, [str(x) for x in list(LossScaler)], self.ui_state, "loss_scaler")
        row += 1

    def __create_layer_frame(self, master, row):
        presets = []
        if self.train_config.model_type.is_stable_diffusion(): #TODO simplify
            presets = sd_presets
        elif self.train_config.model_type.is_stable_diffusion_xl():
            presets = sdxl_presets
        elif self.train_config.model_type.is_stable_diffusion_3():
            presets = sd3_presets
        elif self.train_config.model_type.is_wuerstchen():
            presets = sc_presets
        elif self.train_config.model_type.is_pixart():
            presets = pixart_presets
        elif self.train_config.model_type.is_flux():
            presets = flux_presets
        elif self.train_config.model_type.is_qwen():
            presets = qwen_presets
        elif self.train_config.model_type.is_chroma():
            presets = chroma_presets
        elif self.train_config.model_type.is_sana():
            presets = sana_presets
        elif self.train_config.model_type.is_hunyuan_video():
            presets = hunyuan_video_presets
        elif self.train_config.model_type.is_hi_dream():
            presets = hidream_presets
        else:
            presets = {"full": []}
        components.layer_filter_entry(master, row, 0, self.ui_state,
            preset_var_name="layer_filter_preset", presets=presets,
            preset_label="Layer Filter",
            preset_tooltip="Select a preset defining which layers to train, or select 'Custom' to define your own.\nA blank 'custom' field or 'Full' will train all layers.",
            entry_var_name="layer_filter",
            entry_tooltip="Comma-separated list of diffusion layers to train. Regular expressions (if toggled) are supported. Any model layer with a matching name will be trained",
            regex_var_name="layer_filter_regex",
            regex_tooltip="If enabled, layer filter patterns are interpreted as regular expressions. Otherwise, simple substring matching is used.",
        )


    def __on_layer_filter_preset_change(self):
        if not self.layer_selector:
            return
        selected = self.ui_state.get_var("layer_filter_preset").get()
        self.__preset_set_layer_choice(selected)

    def __hide_layer_entry(self):
        if self.layer_entry and self.layer_entry.winfo_manager():
            self.layer_entry.grid_remove()

    def __show_layer_entry(self):
        if self.layer_entry and not self.layer_entry.winfo_manager():
            self.layer_entry.grid()

    def __open_optimizer_params_window(self):
        window = OptimizerParamsWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __open_scheduler_params_window(self):
        window = SchedulerParamsWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __open_timestep_distribution_window(self):
        window = TimestepDistributionWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __open_offloading_window(self):
        window = OffloadingWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __restore_optimizer_config(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

    def __restore_scheduler_config(self, variable):
        if not hasattr(self, 'lr_scheduler_adv_comp'):
            return

        if variable == "CUSTOM":
            self.lr_scheduler_adv_comp.configure(state="normal")
        else:
            self.lr_scheduler_adv_comp.configure(state="disabled")
