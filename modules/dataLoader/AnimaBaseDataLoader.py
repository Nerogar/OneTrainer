import os

import torch

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.AnimaModel import PROMPT_MAX_LENGTH, AnimaModel
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseAnimaSetup import BaseAnimaSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeAnimaText import EncodeAnimaText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


class _PreparedEncodeVAE(EncodeVAE):
    def __init__(self, *args, prepare_fun, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_fun = prepare_fun

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        self.prepare_fun()
        return super().get_item(variation, index, requested_name)


class _PreparedEncodeAnimaText(EncodeAnimaText):
    def __init__(self, *args, prepare_fun, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_fun = prepare_fun

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        self.prepare_fun()
        return super().get_item(variation, index, requested_name)


@factory.register(BaseDataLoader, ModelType.ANIMA)
class AnimaBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def __prepare_inline_vae(self, config: TrainConfig, model: AnimaModel):
        if not config.image_caching:
            model.vae_to(self.train_device)

    def __vae_dtype_and_autocast_context(self, config: TrainConfig, model: AnimaModel):
        self.__prepare_inline_vae(config, model)

        vae_dtype = next(model.vae.parameters()).dtype
        vae_autocast_context = torch.autocast(device_type=self.train_device.type, enabled=False)
        return vae_dtype, vae_autocast_context

    def __prepare_inline_text_encoder(self, config: TrainConfig, model: AnimaModel):
        if not config.text_caching and not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

    def _preparation_modules(self, config: TrainConfig, model: AnimaModel):
        vae_dtype, vae_autocast_context = self.__vae_dtype_and_autocast_context(config, model)
        self.__prepare_inline_text_encoder(config, model)
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = _PreparedEncodeVAE(
            in_name='image', out_name='latent_image_distribution',
            vae=model.vae, autocast_contexts=[vae_autocast_context], dtype=vae_dtype,
            prepare_fun=lambda: self.__prepare_inline_vae(config, model),
        )
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.125)
        # Anima has no chat template — tokenize raw prompt with both tokenizers
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.tokenizer, max_token_length=PROMPT_MAX_LENGTH)
        tokenize_t5 = Tokenize(in_name='prompt', tokens_out_name='t5_tokens', mask_out_name='t5_tokens_mask', tokenizer=model.t5_tokenizer, max_token_length=PROMPT_MAX_LENGTH)
        # EncodeAnimaText runs Qwen3 encoder + AnimaTextConditioner; output is fixed (512, 1024)
        encode_prompt = _PreparedEncodeAnimaText(
            tokens_name='tokens', tokens_attention_mask_name='tokens_mask',
            t5_tokens_name='t5_tokens', t5_tokens_attention_mask_name='t5_tokens_mask',
            hidden_state_out_name='text_encoder_hidden_state',
            text_encoder=model.text_encoder, text_conditioner=model.text_conditioner,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
            prepare_fun=lambda: self.__prepare_inline_text_encoder(config, model),
        )

        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        modules += [tokenize_prompt, tokenize_t5]

        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)

        return modules

    def _cache_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens', 'tokens_mask', 't5_tokens', 't5_tokens_mask', 'text_encoder_hidden_state',
            'concept'
        ]

        if not config.train_text_encoder_or_embedding():
            text_split_names += ['tokens', 'tokens_mask', 't5_tokens', 't5_tokens_mask', 'text_encoder_hidden_state']

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=config.text_caching and not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: AnimaModel, model_setup: BaseAnimaSetup):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens',
            'tokens_mask',
            't5_tokens',
            't5_tokens_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if not config.train_text_encoder_or_embedding():
            output_names.append('text_encoder_hidden_state')

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: AnimaModel): #TODO clean up
        debug_dir = os.path.join(config.debug_dir, "dataloader")
        vae_dtype, vae_autocast_context = self.__vae_dtype_and_autocast_context(config, model)

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae, autocast_contexts=[vae_autocast_context], dtype=vae_dtype)
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=8)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.tokenizer)

        #FIXME https://github.com/Nerogar/OneTrainer/issues/1015
        #save_image = SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)

        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir, before_save_fun=before_save_fun)

        # These modules don't really work, since they are inserted after a sorting operation that does not include this data
        # SaveImage(image_in_name='mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),

        modules = [decode_image]

        #FIXME https://github.com/Nerogar/OneTrainer/issues/1015
        #modules.append(save_image)

        if config.masked_training or config.model_type.has_mask_input():
            modules += [upscale_mask, save_mask]

        modules += [decode_prompt, save_prompt]

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(self,
            config, model, model_setup, train_progress, is_validation,
            aspect_bucketing_quantization=64,
            allow_video_files=False, #don't allow video files, but...
            vae_frame_dim=True,  #...Anima has a video-capable VAE. convert images to video dimensions
        )
