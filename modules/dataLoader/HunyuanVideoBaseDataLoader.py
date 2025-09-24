import copy
import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.HunyuanVideoModel import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE_CROP_START,
    HunyuanVideoModel,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeClipText import EncodeClipText
from mgds.pipelineModules.EncodeLlamaText import EncodeLlamaText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize
from mgds.pipelineModules.VariationSorting import VariationSorting

import torch


class HunyuanVideoBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: HunyuanVideoModel,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        super().__init__(
            train_device,
            temp_device,
        )

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False

        self.__ds = self.create_dataset(
            config=config,
            model=model,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    def _preparation_modules(self, config: TrainConfig, model: HunyuanVideoModel):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.125)
        add_embeddings_to_prompt_1 = MapData(in_name='prompt', out_name='prompt_1', map_fn=model.add_text_encoder_1_embeddings_to_prompt)
        add_embeddings_to_prompt_2 = MapData(in_name='prompt', out_name='prompt_2', map_fn=model.add_text_encoder_2_embeddings_to_prompt)
        tokenize_prompt_1 = Tokenize(in_name='prompt_1', tokens_out_name='tokens_1', mask_out_name='tokens_mask_1', tokenizer=model.tokenizer_1, max_token_length=77, format_text=DEFAULT_PROMPT_TEMPLATE, additional_format_text_tokens=DEFAULT_PROMPT_TEMPLATE_CROP_START)
        tokenize_prompt_2 = Tokenize(in_name='prompt_2', tokens_out_name='tokens_2', mask_out_name='tokens_mask_2', tokenizer=model.tokenizer_2, max_token_length=77)
        encode_prompt_1 = EncodeLlamaText(tokens_name='tokens_1', tokens_attention_mask_in_name='tokens_mask_1', hidden_state_out_name='text_encoder_1_hidden_state', tokens_attention_mask_out_name='tokens_mask_1', text_encoder=model.text_encoder_1, hidden_state_output_index=-(1 + config.text_encoder_2_layer_skip), autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(), crop_start=DEFAULT_PROMPT_TEMPLATE_CROP_START)
        encode_prompt_2 = EncodeClipText(in_name='tokens_2', tokens_attention_mask_in_name=None, hidden_state_out_name='text_encoder_2_hidden_states', pooled_out_name='text_encoder_2_pooled_state', add_layer_norm=False, text_encoder=model.text_encoder_2, hidden_state_output_index=-(2 + config.text_encoder_layer_skip), autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())

        modules = [rescale_image, encode_image, image_sample]

        if model.tokenizer_1:
            modules.append(add_embeddings_to_prompt_1)
            modules.append(tokenize_prompt_1)
        if model.tokenizer_2:
            modules.append(add_embeddings_to_prompt_2)
            modules.append(tokenize_prompt_2)

        if config.masked_training:
            modules.append(downscale_mask)

        if not config.train_text_encoder_or_embedding() and model.text_encoder_1:
            modules.append(encode_prompt_1)

        if not config.train_text_encoder_2_or_embedding() and model.text_encoder_2:
            modules.append(encode_prompt_2)

        return modules

    def _cache_modules(self, config: TrainConfig, model: HunyuanVideoModel):
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            image_split_names.append('latent_conditioning_image')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt_1', 'tokens_1', 'tokens_mask_1', 'text_encoder_1_hidden_state',
            'prompt_2', 'tokens_2', 'tokens_mask_2', 'text_encoder_2_pooled_state',
            'concept'
        ]

        if not config.train_text_encoder_or_embedding():
            text_split_names.append('tokens_1')
            text_split_names.append('tokens_mask_1')
            text_split_names.append('text_encoder_1_hidden_state')

        if not config.train_text_encoder_2_or_embedding():
            text_split_names.append('tokens_2')
            text_split_names.append('tokens_mask_2')
            text_split_names.append('text_encoder_2_pooled_state')

        image_cache_dir = os.path.join(config.cache_dir, "image")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        def before_cache_text_fun():
            model.to(self.temp_device)

            if not config.train_text_encoder_or_embedding():
                model.text_encoder_1_to(self.train_device)

            if not config.train_text_encoder_2_or_embedding():
                model.text_encoder_2_to(self.train_device)

            model.eval()
            torch_gc()

        image_disk_cache = DiskCache(cache_dir=image_cache_dir, split_names=image_split_names, aggregate_names=image_aggregate_names, variations_in_name='concept.image_variations', balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_image_fun)

        text_disk_cache = DiskCache(cache_dir=text_cache_dir, split_names=text_split_names, aggregate_names=[], variations_in_name='concept.text_variations', balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_text_fun)

        modules = []

        if config.latent_caching:
            modules.append(image_disk_cache)

        if config.latent_caching:
            sort_names = [x for x in sort_names if x not in image_aggregate_names]
            sort_names = [x for x in sort_names if x not in image_split_names]

            if not config.train_text_encoder_or_embedding() or not config.train_text_encoder_2_or_embedding():
                modules.append(text_disk_cache)
                sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(names=sort_names, balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], group_enabled_in_name='concept.enabled')
            modules.append(variation_sorting)

        return modules

    def _output_modules(self, config: TrainConfig, model: HunyuanVideoModel):
        output_names = [
            'image_path', 'latent_image',
            'prompt_1', 'prompt_2',
            'tokens_1', 'tokens_2',
            'tokens_mask_1', 'tokens_mask_2',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            output_names.append('latent_conditioning_image')

        if not config.train_text_encoder_or_embedding():
            output_names.append('text_encoder_1_hidden_state')

        if not config.train_text_encoder_2_or_embedding():
            output_names.append('text_encoder_2_pooled_state')

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.vae_to(self.train_device)
            model.eval()
            torch_gc()

        return self._output_modules_from_out_names(
            output_names=output_names,
            config=config,
            before_cache_image_fun=before_cache_image_fun,
            use_conditioning_image=True,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: HunyuanVideoModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=8)
        decode_prompt = DecodeTokens(in_name='tokens_1', out_name='decoded_prompt', tokenizer=model.tokenizer_1)

        #FIXME https://github.com/Nerogar/OneTrainer/issues/1015
        #save_image = SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        #save_conditioning_image = SaveImage(image_in_name='decoded_conditioning_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)

        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir, before_save_fun=before_save_fun)

        # These modules don't really work, since they are inserted after a sorting operation that does not include this data
        # SaveImage(image_in_name='mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),

        modules = []

        modules.append(decode_image)

        #FIXME https://github.com/Nerogar/OneTrainer/issues/1015
        #modules.append(save_image)

        #if config.model_type.has_conditioning_image_input():
        #    modules.append(save_conditioning_image)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(upscale_mask)
            modules.append(save_mask)

        modules.append(decode_prompt)
        modules.append(save_prompt)

        return modules

    def create_dataset(
            self,
            config: TrainConfig,
            model: HunyuanVideoModel,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        enumerate_input = self._enumerate_input_modules(config, allow_videos=True)
        load_input = self._load_input_modules(config, model.train_dtype, allow_video=True)
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, 64, True)
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
        inpainting_modules = self._inpainting_modules(config)
        preparation_modules = self._preparation_modules(config, model)
        cache_modules = self._cache_modules(config, model)
        output_modules = self._output_modules(config, model)

        debug_modules = self._debug_modules(config, model)

        return self._create_mgds(
            config,
            [
                enumerate_input,
                load_input,
                mask_augmentation,
                aspect_bucketing_in,
                crop_modules,
                augmentation_modules,
                inpainting_modules,
                preparation_modules,
                cache_modules,
                output_modules,

                debug_modules if config.debug_mode else None,
                # inserted before output_modules, which contains a sorting operation
            ],
            train_progress,
            is_validation
        )
