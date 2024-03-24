import customtkinter as ctk

from modules.ui.OptimizerParamsWindow import OptimizerParamsWindow
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import change_optimizer
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class TrainingTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super(TrainingTab, self).__init__()

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
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(column_0, column_1, column_2)
        elif self.train_config.model_type.is_pixart_alpha():
            self.__setup_pixart_alpha_ui(column_0, column_1, column_2)

    def __setup_stable_diffusion_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)

        self.__create_base2_frame(column_1, 0)
        self.__create_unet_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2, supports_vb_loss=False)

    def __setup_stable_diffusion_xl_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_1_frame(column_0, 1)
        self.__create_text_encoder_2_frame(column_0, 2)

        self.__create_base2_frame(column_1, 0)
        self.__create_unet_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2, supports_vb_loss=False)

    def __setup_wuerstchen_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)

        self.__create_base2_frame(column_1, 0)
        self.__create_prior_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_masked_frame(column_2, 0)
        self.__create_loss_frame(column_2, 1, supports_vb_loss=False)

    def __setup_pixart_alpha_ui(self, column_0, column_1, column_2):
        self.__create_base_frame(column_0, 0)
        self.__create_text_encoder_frame(column_0, 1)

        self.__create_base2_frame(column_1, 0)
        self.__create_prior_frame(column_1, 1)
        self.__create_noise_frame(column_1, 2)

        self.__create_align_prop_frame(column_2, 0)
        self.__create_masked_frame(column_2, 1)
        self.__create_loss_frame(column_2, 2, supports_vb_loss=True)

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
        components.label(frame, 1, 0, "Learning Rate Scheduler",
                         tooltip="Learning rate scheduler that automatically changes the learning rate during training")
        components.options(frame, 1, 1, [str(x) for x in list(LearningRateScheduler)], self.ui_state,
                           "learning_rate_scheduler")

        # learning rate
        components.label(frame, 2, 0, "Learning Rate",
                         tooltip="The base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "learning_rate")

        # learning rate warmup steps
        components.label(frame, 3, 0, "Learning Rate Warmup Steps",
                         tooltip="The number of steps it takes to gradually increase the learning rate from 0 to the specified learning rate")
        components.entry(frame, 3, 1, self.ui_state, "learning_rate_warmup_steps")

        # learning rate cycles
        components.label(frame, 4, 0, "Learning Rate Cycles",
                         tooltip="The number of learning rate cycles. This is only applicable if the learning rate scheduler supports cycles")
        components.entry(frame, 4, 1, self.ui_state, "learning_rate_cycles")

        # epochs
        components.label(frame, 5, 0, "Epochs",
                         tooltip="The number of epochs for a full training run")
        components.entry(frame, 5, 1, self.ui_state, "epochs")

        # batch size
        components.label(frame, 6, 0, "Batch Size",
                         tooltip="The batch size of one training step")
        components.entry(frame, 6, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(frame, 7, 0, "Accumulation Steps",
                         tooltip="Number of accumulation steps. Increase this number to trade batch size for training speed")
        components.entry(frame, 7, 1, self.ui_state, "gradient_accumulation_steps")

        # Learning Rate Scaler
        components.label(frame, 8, 0, "Learning Rate Scaler",
                         tooltip="Selects the type of learning rate scaling to use during training. Functionally equated as: LR * SQRT(selection)")
        components.options(frame, 8, 1, [str(x) for x in list(LearningRateScaler)], self.ui_state,
                           "learning_rate_scaler")

    def __create_base2_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # attention mechanism
        components.label(frame, 0, 0, "Attention",
                         tooltip="The attention mechanism used during training. This has a big effect on speed and memory consumption")
        components.options(frame, 0, 1, [str(x) for x in list(AttentionMechanism)], self.ui_state,
                           "attention_mechanism")

        # ema
        components.label(frame, 1, 0, "EMA",
                         tooltip="EMA averages the training progress over many steps, better preserving different concepts in big datasets")
        components.options(frame, 1, 1, [str(x) for x in list(EMAMode)], self.ui_state,
                           "ema")

        # ema decay
        components.label(frame, 2, 0, "EMA Decay",
                         tooltip="Decay parameter of the EMA model. Higher numbers will average more steps. For datasets of hundreds or thousands of images, set this to 0.9999. For smaller datasets, set it to 0.999 or even 0.998")
        components.entry(frame, 2, 1, self.ui_state, "ema_decay")

        # ema update step interval
        components.label(frame, 3, 0, "EMA Update Step Interval",
                         tooltip="Number of steps between EMA update steps")
        components.entry(frame, 3, 1, self.ui_state, "ema_update_step_interval")

        # gradient checkpointing
        components.label(frame, 4, 0, "Gradient checkpointing",
                         tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        components.switch(frame, 4, 1, self.ui_state, "gradient_checkpointing")

        # train dtype
        components.label(frame, 5, 0, "Train Data Type",
                         tooltip="The mixed precision data type used for training. This can increase training speed, but reduces precision")
        components.options_kv(frame, 5, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], self.ui_state, "train_dtype")

        # fallback train dtype
        components.label(frame, 6, 0, "Fallback Train Data Type",
                         tooltip="The mixed precision data type used for training stages that don't support float16 data types. This can increase training speed, but reduces precision")
        components.options_kv(frame, 6, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "fallback_train_dtype")

        # autocast cache
        components.label(frame, 7, 0, "Autocast Cache",
                         tooltip="Enables the autocast cache. Disabling this reduces memory usage, but increases training time")
        components.switch(frame, 7, 1, self.ui_state, "enable_autocast_cache")


        # resolution
        components.label(frame, 8, 0, "Resolution",
                         tooltip="The resolution used for training. Optionally specify multiple resolutions separated by a comma.")
        components.entry(frame, 8, 1, self.ui_state, "resolution")

    def __create_align_prop_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # align prop
        components.label(frame, 0, 0, "AlignProp",
                         tooltip="Enables AlignProp training")
        components.switch(frame, 0, 1, self.ui_state, "align_prop")

        # align prop probability
        components.label(frame, 1, 0, "AlignProp Probability",
                         tooltip="When AlignProp is enabled, specifies the number of training steps done using AlignProp calculations")
        components.entry(frame, 1, 1, self.ui_state, "align_prop_probability")

        # align prop loss
        components.label(frame, 2, 0, "AlignProp Loss",
                         tooltip="Specifies the loss function used for AlignProp calculations")
        components.options(frame, 2, 1, [str(x) for x in list(AlignPropLoss)], self.ui_state, "align_prop_loss")

        # align prop weight
        components.label(frame, 3, 0, "AlignProp Weight",
                         tooltip="A weight multiplier for the AlignProp loss")
        components.entry(frame, 3, 1, self.ui_state, "align_prop_weight")

        # align prop steps
        components.label(frame, 4, 0, "AlignProp Steps",
                         tooltip="Number of inference steps for each AlignProp step")
        components.entry(frame, 4, 1, self.ui_state, "align_prop_steps")

        # align prop truncate steps
        components.label(frame, 5, 0, "AlignProp Truncate Steps",
                         tooltip="Fraction of steps to randomly truncate when using AlignProp. This is needed to increase model diversity.")
        components.entry(frame, 5, 1, self.ui_state, "align_prop_truncate_steps")

        # align prop truncate steps
        components.label(frame, 6, 0, "AlignProp CFG Scale",
                         tooltip="CFG Scale for inference steps of AlignProp calculations")
        components.entry(frame, 6, 1, self.ui_state, "align_prop_cfg_scale")

    def __create_text_encoder_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train text encoder
        components.label(frame, 0, 0, "Train Text Encoder",
                         tooltip="Enables training the text encoder model")
        components.switch(frame, 0, 1, self.ui_state, "text_encoder.train")

        # train text encoder epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder")
        components.time_entry(frame, 1, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)

        # text encoder learning rate
        components.label(frame, 2, 0, "Text Encoder Learning Rate",
                         tooltip="The learning rate of the text encoder. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "text_encoder.learning_rate")

        # text encoder layer skip (clip skip)
        components.label(frame, 3, 0, "Clip Skip",
                         tooltip="The number of clip layers to skip. 0 = disabled")
        components.entry(frame, 3, 1, self.ui_state, "text_encoder_layer_skip")

    def __create_text_encoder_1_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train text encoder
        components.label(frame, 0, 0, "Train Text Encoder 1",
                         tooltip="Enables training the text encoder 1 model")
        components.switch(frame, 0, 1, self.ui_state, "text_encoder.train")

        # train text encoder epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 1")
        components.time_entry(frame, 1, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)

        # text encoder learning rate
        components.label(frame, 2, 0, "Text Encoder 1 Learning Rate",
                         tooltip="The learning rate of the text encoder 1. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "text_encoder.learning_rate")

        # text encoder layer skip (clip skip)
        components.label(frame, 3, 0, "Clip Skip 1",
                         tooltip="The number of clip layers to skip. 0 = disabled")
        components.entry(frame, 3, 1, self.ui_state, "text_encoder_layer_skip")

    def __create_text_encoder_2_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # train text encoder
        components.label(frame, 0, 0, "Train Text Encoder 2",
                         tooltip="Enables training the text encoder 2 model")
        components.switch(frame, 0, 1, self.ui_state, "text_encoder_2.train")

        # train text encoder epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 1")
        components.time_entry(frame, 1, 1, self.ui_state, "text_encoder_2.stop_training_after",
                              "text_encoder_2.stop_training_after_unit", supports_time_units=False)

        # text encoder learning rate
        components.label(frame, 2, 0, "Text Encoder 2 Learning Rate",
                         tooltip="The learning rate of the text encoder 2. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "text_encoder_2.learning_rate")

        # text encoder layer skip (clip skip)
        components.label(frame, 3, 0, "Clip Skip 2",
                         tooltip="The number of clip layers to skip. 0 = disabled")
        components.entry(frame, 3, 1, self.ui_state, "text_encoder_2_layer_skip")

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
        components.label(frame, 3, 0, "Rescale Noise Scheduler",
                         tooltip="Rescales the noise scheduler to a zero terminal signal to noise ratio and switches the model to a v-prediction target")
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

    def __create_noise_frame(self, master, row):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # offset noise weight
        components.label(frame, 0, 0, "Offset Noise Weight",
                         tooltip="The weight of offset noise added to each training step")
        components.entry(frame, 0, 1, self.ui_state, "offset_noise_weight")

        # perturbation noise weight
        components.label(frame, 1, 0, "Perturbation Noise Weight",
                         tooltip="The weight of perturbation noise added to each training step")
        components.entry(frame, 1, 1, self.ui_state, "perturbation_noise_weight")

        # min noising strength
        components.label(frame, 2, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        components.entry(frame, 2, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(frame, 3, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        components.entry(frame, 3, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(frame, 4, 0, "Noising Weight",
                         tooltip="Controls the emphasis on certain noise levels during training. A value of 0 disables this feature, leading to a uniform distribution where all noise levels are equally likely. Positive values increase the likelihood of selecting higher noise levels at any particular training step (resulting in smoother, low-frequency details), while negative values favor lower noise levels (sharper, high-frequency details). Note that extreme values (like -10 or 10) create a strong emphasis, significantly changing the focus of training.\n\nThe distribution function is as follows for a noise strength from 0 to 1: chance(noise_strength) = 1 / (1 + exp(-noising_weight * (noise_strength - noising_bias)))",
                         wide_tooltip=True)
        components.entry(frame, 4, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(frame, 5, 0, "Noising Bias",
                         tooltip="Adjusts the balance point in the noise level selection process. The setting ranges from 0 to 1. At 0.5, the selection is balanced, neither favoring lower nor higher noise levels. Lower values shift the distribution curve left towards training lower noise levels (fine details), while higher values shift the distribution towards high noise levels (low-frequency details or image composition).\n\nThe distribution function is as follows for a noise strength from 0 to 1: chance(noise_strength) = 1 / (1 + exp(-noising_weight * (noise_strength - noising_bias)))",
                         wide_tooltip=True)
        components.entry(frame, 5, 1, self.ui_state, "noising_bias")

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

    def __create_loss_frame(self, master, row, supports_vb_loss: bool):
        frame = ctk.CTkFrame(master=master, corner_radius=5)
        frame.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        # MSE Strength
        components.label(frame, 0, 0, "MSE Strength",
                         tooltip="Mean Squared Error strength for custom loss settings. MAE + MSE Strengths generally should sum to 1.")
        components.entry(frame, 0, 1, self.ui_state, "mse_strength")

        # MAE Strength
        components.label(frame, 1, 0, "MAE Strength",
                         tooltip="Mean Absolute Error strength for custom loss settings. MAE + MSE Strengths generally should sum to 1.")
        components.entry(frame, 1, 1, self.ui_state, "mae_strength")

        if supports_vb_loss:
            # VB Strength
            components.label(frame, 2, 0, "VB Strength",
                             tooltip="Variational lower-bound strength for custom loss settings. Should be set to 1 for variational diffusion models")
            components.entry(frame, 2, 1, self.ui_state, "vb_loss_strength")

        # Loss Weight function
        components.label(frame, 3, 0, "Loss Weight Function",
                         tooltip="Choice of loss weight function. Can help the model learn details more accurately.")
        components.options(frame, 3, 1, [str(x) for x in list(LossWeight)], self.ui_state, "loss_weight_fn")
        
        # Loss weight strength
        components.label(frame, 4, 0, "Gamma",
                         tooltip="Inverse strength of loss weighting. Range: 1-20, only applies to Min SNR and P2.")
        components.entry(frame, 4, 1, self.ui_state, "loss_weight_strength")

        # Loss Scaler
        components.label(frame, 5, 0, "Loss Scaler",
                         tooltip="Selects the type of loss scaling to use during training. Functionally equated as: Loss * selection")
        components.options(frame, 5, 1, [str(x) for x in list(LossScaler)], self.ui_state, "loss_scaler")

    def __open_optimizer_params_window(self):
        window = OptimizerParamsWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def __restore_optimizer_config(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)
