import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.LTXModel import PROMPT_MAX_LENGTH, PROMPT_PADDING_SIDE, LTXModel
from modules.modelSetup.BaseLTXSetup import BaseLTXSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeGemma3Text import EncodeGemma3Text
from mgds.pipelineModules.EncodeLTX2Connectors import EncodeLTX2Connectors
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.SaveVideo import SaveVideo
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


@factory.register(BaseDataLoader, ModelType.LTX_2)
class LTXBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: LTXModel):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        # LTX VAE is 32x spatial (not the usual 8x), so the mask latent scales by 1/32, not 1/8
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=1/32)
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.tokenizer, max_token_length=PROMPT_MAX_LENGTH, padding_side=PROMPT_PADDING_SIDE)
        encode_prompt = EncodeGemma3Text(tokens_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', tokens_attention_mask_out_name='tokens_mask',
                                          text_encoder=model.text_encoder, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        # cache the connectors' small per-modality output instead of the huge stacked TE output: ~30x smaller
        # text cache and no connector forward at train time. No Prune/Pad - the connector output is dense
        # fixed-length, since learnable registers replace padding.
        encode_connectors = EncodeLTX2Connectors(
            hidden_state_in_name='text_encoder_hidden_state', tokens_attention_mask_in_name='tokens_mask',
            video_embeds_out_name='connector_video_embeds', audio_embeds_out_name='connector_audio_embeds',
            connectors=model.connectors, padding_side=PROMPT_PADDING_SIDE, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())

        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)
        modules += [tokenize_prompt, encode_prompt, encode_connectors]

        return modules

    def _cache_modules(self, config: TrainConfig, model: LTXModel, model_setup: BaseLTXSetup):
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens', 'tokens_mask', 'connector_video_embeds',
            'concept'
        ]

        text_split_names += ['tokens', 'tokens_mask', 'connector_video_embeds']

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=True,
        )

    def _output_modules(self, config: TrainConfig, model: LTXModel, model_setup: BaseLTXSetup):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens',
            'tokens_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
            'connector_video_embeds',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: LTXModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.materialize_only("vae")

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=32)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.tokenizer)

        # SaveVideo instead of SaveImage: latents here are 5D (vae_frame_dim), so the decode is [C, F, H, W]
        # and SaveImage's ToPILImage can't take it (the #1015 FIXME the other video loaders still carry).
        save_video = SaveVideo(video_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, fps=model.NATIVE_FPS, before_save_fun=before_save_fun)

        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir, before_save_fun=before_save_fun)

        modules = [decode_image, save_video]

        if config.masked_training or config.model_type.has_mask_input():
            modules += [upscale_mask, save_mask]

        modules += [decode_prompt, save_prompt]

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: LTXModel,
            model_setup: BaseLTXSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(self,
            config, model, model_setup, train_progress, is_validation,
            aspect_bucketing_quantization=64,
            frame_dim_enabled=True,
            allow_video_files=True,
            vae_frame_dim=True,
        )
