import copy
import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.wuerstchen.EncodeWuerstchenEffnet import EncodeWuerstchenEffnet
from modules.model.WuerstchenModel import WuerstchenModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader
from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeClipText import EncodeClipText
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.NormalizeImageChannels import NormalizeImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize
from mgds.pipelineModules.VariationSorting import VariationSorting

import torch


class WuerstchenBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: WuerstchenModel,
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

    def _preparation_modules(self, config: TrainConfig, model: WuerstchenModel):
        downscale_image = ScaleImage(in_name='image', out_name='image', factor=0.75)
        normalize_image = NormalizeImageChannels(image_in_name='image', image_out_name='image', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        encode_image = EncodeWuerstchenEffnet(in_name='image', out_name='latent_image', effnet_encoder=model.effnet_encoder, autocast_contexts=[model.autocast_context, model.effnet_encoder_autocast_context], dtype=model.effnet_encoder_train_dtype.torch_dtype())
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.75 * 0.03125) # *0.75 / 32.0
        add_embeddings_to_prompt = MapData(in_name='prompt', out_name='prompt', map_fn=model.add_prior_text_encoder_embeddings_to_prompt)
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.prior_tokenizer, max_token_length=model.prior_tokenizer.model_max_length)
        if model.model_type.is_wuerstchen_v2():
            encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name=None, add_layer_norm=True, text_encoder=model.prior_text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        elif model.model_type.is_stable_cascade():
            encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name='pooled_text_encoder_output', add_layer_norm=False, text_encoder=model.prior_text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())

        modules = [
            downscale_image, normalize_image, encode_image,
            add_embeddings_to_prompt, tokenize_prompt,
        ]

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)

        return modules

    def _cache_modules(self, config: TrainConfig, model: WuerstchenModel):
        image_split_names = [
            'latent_image',
            'original_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = ['tokens', 'tokens_mask', 'text_encoder_hidden_state']
        if model.model_type.is_stable_cascade():
            text_split_names.append('pooled_text_encoder_output')

        sort_names = text_split_names + image_aggregate_names + image_split_names + [
            'prompt', 'concept'
        ]

        image_cache_dir = os.path.join(config.cache_dir, "image")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.effnet_encoder_to(self.train_device)
            model.eval()
            torch_gc()

        def before_cache_text_fun():
            model.to(self.temp_device)
            model.prior_text_encoder_to(self.train_device)
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

            if not config.train_text_encoder_or_embedding():
                modules.append(text_disk_cache)
                sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(names=sort_names, balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], group_enabled_in_name='concept.enabled')
            modules.append(variation_sorting)

        return modules

    def _output_modules(self, config: TrainConfig, model: WuerstchenModel):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens',
            'tokens_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if not config.train_text_encoder_or_embedding():
            output_names.append('text_encoder_hidden_state')

            if model.model_type.is_stable_cascade():
                output_names.append('pooled_text_encoder_output')

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.effnet_encoder_to(self.train_device)
            model.eval()
            torch_gc()

        return self._output_modules_from_out_names(
            output_names=output_names,
            config=config,
            before_cache_image_fun=before_cache_image_fun,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: WuerstchenModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=1.0/0.75)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.prior_tokenizer)
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir)

        # These modules don't really work, since they are inserted after a sorting operation that does not include this data
        # SaveImage(image_in_name='mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),

        modules = []

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(upscale_mask)
            modules.append(save_mask)

        modules.append(decode_prompt)
        modules.append(save_prompt)

        return modules

    def create_dataset(
            self,
            config: TrainConfig,
            model: WuerstchenModel,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        enumerate_input = self._enumerate_input_modules(config)
        load_input = self._load_input_modules(config, model.train_dtype)
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config, 128)
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
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
                preparation_modules,
                cache_modules,
                output_modules,

                debug_modules if config.debug_mode else None,
                # inserted before output_modules, which contains a sorting operation
            ],
            train_progress,
            is_validation,
        )
