from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.WuerstchenModel import WuerstchenModel, WuerstchenModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionMixin import ModelSetupDiffusionMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_clip_encoder_layers,
    enable_checkpointing_for_stable_cascade_blocks,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import (
    create_autocast_context,
    disable_bf16_on_fp16_autocast_context,
    disable_fp16_autocast_context,
)
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseWuerstchenSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupDiffusionMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    # This is correct for the latest cascade, but other Wuerstchen models may have
    # different names. I honestly don't know what makes a good preset here so I'm
    # just guessing.
    LAYER_PRESETS = {
        "attn-only": ["attention"],
        "full": [],
        "down-blocks": ["down_blocks"],
        "up-blocks": ["up_blocks"],
        "mapper-only": ["mapper"],
    }

    def setup_optimizations(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            if model.model_type.is_wuerstchen_v2():
                model.prior_prior.enable_gradient_checkpointing()
                enable_checkpointing_for_clip_encoder_layers(model.prior_text_encoder, config)
            elif model.model_type.is_stable_cascade():
                enable_checkpointing_for_stable_cascade_blocks(model.prior_prior, config)
                enable_checkpointing_for_clip_encoder_layers(model.prior_text_encoder, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.decoder_vqgan)
            apply_circular_padding_to_conv2d(model.decoder_decoder)
            apply_circular_padding_to_conv2d(model.prior_prior)
            if model.prior_prior_lora is not None:
                apply_circular_padding_to_conv2d(model.prior_prior_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().decoder_text_encoder,
            config.weight_dtypes().decoder,
            config.weight_dtypes().decoder_vqgan,
            config.weight_dtypes().effnet_encoder,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().prior,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        if model.model_type.is_stable_cascade():
            model.prior_autocast_context, model.prior_train_dtype = disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().prior,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
                config.enable_autocast_cache,
            )
        else:
            model.prior_train_dtype = model.train_dtype

        model.effnet_encoder_autocast_context, model.effnet_encoder_train_dtype = disable_bf16_on_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            [
                config.weight_dtypes().effnet_encoder,
            ],
            config.enable_autocast_cache,
        )

        if model.model_type.is_wuerstchen_v2():
            quantize_layers(model.decoder_text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.decoder_decoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.decoder_vqgan, self.train_device, model.train_dtype, config)
        quantize_layers(model.effnet_encoder, self.train_device, model.effnet_encoder_train_dtype, config)
        quantize_layers(model.prior_text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.prior_prior, self.train_device, model.prior_train_dtype, config)

    def _setup_embeddings(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                embedding_state = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.prior_tokenizer,
                    model.prior_text_encoder,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[0][0][1:],
                )
            else:
                embedding_state = embedding_state.get("clip_g_out", embedding_state.get("clip_g", None))

            embedding_state = embedding_state.to(
                dtype=model.prior_text_encoder.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = WuerstchenModelEmbedding(
                embedding_config.uuid,
                embedding_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

            model.additional_embeddings = additional_embeddings

            self._add_embeddings_to_tokenizer(model.prior_tokenizer, model.all_prior_text_encoder_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        model.prior_embedding_wrapper = AdditionalEmbeddingWrapper(
            tokenizer=model.prior_tokenizer,
            orig_module=model.prior_text_encoder.text_model.embeddings.token_embedding,
            embeddings=model.all_prior_text_encoder_embeddings(),
        )
        model.prior_embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_prior_text_encoder_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding = embedding_config.train and \
                              not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding)

    def __alpha_cumprod(
            self,
            timesteps: Tensor,
            dim: int,
    ):
        # copied and modified from https://github.com/dome272/wuerstchen
        s = torch.tensor([0.008], device=timesteps.device, dtype=torch.float32)
        init_alpha_cumprod = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        alpha_cumprod = torch.cos((timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2 / init_alpha_cumprod
        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        alpha_cumprod = alpha_cumprod.view(timesteps.shape[0])
        while alpha_cumprod.dim() < dim:
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)
        return alpha_cumprod

    def predict(
            self,
            model: WuerstchenModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            latent_image = batch['latent_image']
            if model.model_type.is_wuerstchen_v2():
                scaled_latent_image = latent_image.add(1.0).div(42.0)
            elif model.model_type.is_stable_cascade():
                scaled_latent_image = latent_image

            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            timestep = self._get_timestep_continuous(
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
            )

            if model.model_type.is_wuerstchen_v2():
                timestep = timestep.mul(1.08).add(0.001).clamp(0.001, 1.0)
            elif model.model_type.is_stable_cascade():
                timestep = timestep.add(0.001).clamp(0.001, 1.0)

            scaled_noisy_latent_image = self._add_noise_continuous(
                scaled_latent_image,
                latent_noise,
                timestep,
                self.__alpha_cumprod,
            )

            text_embedding, pooled_text_text_embedding = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch['tokens'],
                tokens_mask=batch['tokens_mask'],
                text_encoder_layer_skip=config.text_encoder_layer_skip,
                text_encoder_output=batch[
                    'text_encoder_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                pooled_text_encoder_output=batch[
                    'pooled_text_encoder_output'] if not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            latent_input = scaled_noisy_latent_image

            if model.model_type.is_wuerstchen_v2():
                prior_kwargs = {
                    'c': text_embedding.to(dtype=model.prior_train_dtype.torch_dtype()),
                }
            elif model.model_type.is_stable_cascade():
                clip_img = torch.zeros(
                    size=(text_embedding.shape[0], 1, 768),
                    dtype=model.prior_train_dtype.torch_dtype(),
                    device=self.train_device,
                )
                prior_kwargs = {
                    'clip_text': text_embedding.to(dtype=model.prior_train_dtype.torch_dtype()),
                    'clip_text_pooled': pooled_text_text_embedding.to(dtype=model.prior_train_dtype.torch_dtype()),
                    'clip_img': clip_img,
                }

            with model.prior_autocast_context:
                predicted_latent_noise = model.prior_prior(
                    latent_input.to(dtype=model.prior_train_dtype.torch_dtype()),
                    timestep.to(dtype=model.prior_train_dtype.torch_dtype()),
                    **prior_kwargs,
                )
                if model.model_type.is_stable_cascade():
                    predicted_latent_noise = predicted_latent_noise.sample

            model_output_data = {
                'loss_type': 'target',
                'predicted': predicted_latent_noise,
                'prediction_type': 'epsilon',  # the DDPMWuerstchenScheduler only supports eps prediction
                'target': latent_noise,
                'timestep': timestep,
            }

            if config.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.prior_tokenizer),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    # noise
                    self._save_image(
                        self._project_latent_to_image(latent_noise).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # predicted noise
                    self._save_image(
                        self._project_latent_to_image(predicted_latent_noise).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step
                    )

                    # noisy image
                    self._save_image(
                        self._project_latent_to_image(scaled_noisy_latent_image).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "3-noisy_image",
                        train_progress.global_step
                    )

                    # predicted image
                    alpha_cumprod = self.__alpha_cumprod(timestep, latent_noise.dim())
                    sqrt_alpha_prod = alpha_cumprod ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                        / sqrt_alpha_prod
                    self._save_image(
                        self._project_latent_to_image(scaled_predicted_latent_image).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step
                    )

                    # image
                    self._save_image(
                        self._project_latent_to_image(scaled_latent_image).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step
                    )

        return model_output_data

    def calculate_loss(
            self,
            model: WuerstchenModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._diffusion_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            alphas_cumprod_fun=self.__alpha_cumprod,
        ).mean()

    def prepare_text_caching(self, model: WuerstchenModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
