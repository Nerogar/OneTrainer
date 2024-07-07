import os
import re

import torch
from mgds.MGDS import TrainDataLoader, MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeClipText import EncodeClipText
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.GetFilename import GetFilename
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.NormalizeImageChannels import NormalizeImageChannels
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.ReplaceText import ReplaceText
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.SelectInput import SelectInput
from mgds.pipelineModules.SelectRandomText import SelectRandomText
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModules.Tokenize import Tokenize
from mgds.pipelineModules.VariationSorting import VariationSorting

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.wuerstchen.EncodeWuerstchenEffnet import EncodeWuerstchenEffnet
from modules.model.WuerstchenModel import WuerstchenModel
from modules.util import path_util
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.torch_util import torch_gc


class WuerstchenBaseDataLoader(BaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: WuerstchenModel,
            train_progress: TrainProgress,
    ):
        super(WuerstchenBaseDataLoader, self).__init__(
            train_device,
            temp_device,
        )

        self.__ds = self.create_dataset(
            config=config,
            model=model,
            train_progress=train_progress,
        )
        self.__dl = TrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    def _enumerate_input_modules(self, config: TrainConfig) -> list:
        supported_extensions = path_util.supported_image_extensions()

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', path_out_name='image_path', concept_out_name='concept',
            extensions=supported_extensions, include_postfix=None, exclude_postfix=['-masklabel'], include_subdirectories_in_name='concept.include_subdirectories'
        )

        mask_path = ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png')
        sample_prompt_path = ModifyPath(in_name='image_path', out_name='sample_prompt_path', postfix='', extension='.txt')

        modules = [collect_paths, sample_prompt_path]

        if config.masked_training:
            modules.append(mask_path)

        return modules


    def _load_input_modules(self, config: TrainConfig, model: WuerstchenModel) -> list:
        load_image = LoadImage(path_in_name='image_path', image_out_name='image', range_min=0, range_max=1, dtype=model.effnet_encoder_train_dtype.torch_dtype())

        generate_mask = GenerateImageLike(image_in_name='image', image_out_name='mask', color=255, range_min=0, range_max=1, channels=1)
        load_mask = LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1, dtype=model.effnet_encoder_train_dtype.torch_dtype())

        load_sample_prompts = LoadMultipleTexts(path_in_name='sample_prompt_path', texts_out_name='sample_prompts')
        load_concept_prompts = LoadMultipleTexts(path_in_name='concept.text.prompt_path', texts_out_name='concept_prompts')
        filename_prompt = GetFilename(path_in_name='image_path', filename_out_name='filename_prompt', include_extension=False)
        select_prompt_input = SelectInput(setting_name='concept.text.prompt_source', out_name='prompts', setting_to_in_name_map={
            'sample': 'sample_prompts',
            'concept': 'concept_prompts',
            'filename': 'filename_prompt',
        }, default_in_name='sample_prompts')
        select_random_text = SelectRandomText(texts_in_name='prompts', text_out_name='prompt')

        replace_embedding_text = []
        for embedding in model.additional_embeddings:
            all_token_string = ''.join(embedding.text_tokens)
            replace_embedding_text.append(ReplaceText(text_in_name='prompt', text_out_name='prompt', old_text=embedding.placeholder, new_text=all_token_string))

        if model.embedding is not None:
            all_token_string = ''.join(model.embedding.text_tokens)
            replace_embedding_text.append(ReplaceText(text_in_name='prompt', text_out_name='prompt', old_text=model.embedding.placeholder, new_text=all_token_string))

        modules = [load_image, load_sample_prompts, load_concept_prompts, filename_prompt, select_prompt_input, select_random_text]

        if config.masked_training:
            modules.append(generate_mask)
            modules.append(load_mask)
        elif config.model_type.has_mask_input():
            modules.append(generate_mask)

        modules.append(replace_embedding_text)

        return modules


    def _mask_augmentation_modules(self, config: TrainConfig) -> list:
        inputs = ['image']

        lowest_resolution = min([int(x.strip()) for x in re.split('\D', config.resolution) if x.strip() != ''])
        circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='concept.image.enable_random_circular_mask_shrink')
        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=inputs, min_size=lowest_resolution, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='concept.image.enable_random_mask_rotate_crop')

        modules = []

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(circular_mask_shrink)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(random_mask_rotate_crop)

        return modules


    def _aspect_bucketing_in(self, config: TrainConfig):
        calc_aspect = CalcAspect(image_in_name='image', resolution_out_name='original_resolution')

        aspect_bucketing = AspectBucketing(
            quantization=128,
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        single_aspect_calculation = SingleAspectCalculation(
            resolution_in_name='original_resolution',
            target_resolution_in_name='settings.target_resolution',
            enable_target_resolutions_override_in_name='concept.image.enable_resolution_override',
            target_resolutions_override_in_name='concept.image.resolution_override',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        modules = [calc_aspect]

        if config.aspect_ratio_bucketing:
            modules.append(aspect_bucketing)
        else:
            modules.append(single_aspect_calculation)

        return modules


    def _crop_modules(self, config: TrainConfig):
        inputs = ['image']

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        scale_crop = ScaleCropImage(names=inputs, scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.image.enable_crop_jitter', crop_offset_out_name='crop_offset')

        modules = [scale_crop]

        return modules


    def _augmentation_modules(self, config: TrainConfig):
        inputs = ['image']

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append('mask')

        random_flip = RandomFlip(names=inputs, enabled_in_name='concept.image.enable_random_flip', fixed_enabled_in_name='concept.image.enable_fixed_flip')
        random_rotate = RandomRotate(names=inputs, enabled_in_name='concept.image.enable_random_rotate', fixed_enabled_in_name='concept.image.enable_fixed_rotate', max_angle_in_name='concept.image.random_rotate_max_angle')
        random_brightness = RandomBrightness(names=['image'], enabled_in_name='concept.image.enable_random_brightness', fixed_enabled_in_name='concept.image.enable_fixed_brightness', max_strength_in_name='concept.image.random_brightness_max_strength')
        random_contrast = RandomContrast(names=['image'], enabled_in_name='concept.image.enable_random_contrast', fixed_enabled_in_name='concept.image.enable_fixed_contrast', max_strength_in_name='concept.image.random_contrast_max_strength')
        random_saturation = RandomSaturation(names=['image'], enabled_in_name='concept.image.enable_random_saturation', fixed_enabled_in_name='concept.image.enable_fixed_saturation', max_strength_in_name='concept.image.random_saturation_max_strength')
        random_hue = RandomHue(names=['image'], enabled_in_name='concept.image.enable_random_hue', fixed_enabled_in_name='concept.image.enable_fixed_hue', max_strength_in_name='concept.image.random_hue_max_strength')
        shuffle_tags = ShuffleTags(text_in_name='prompt', enabled_in_name='concept.text.enable_tag_shuffling', delimiter_in_name='concept.text.tag_delimiter', keep_tags_count_in_name='concept.text.keep_tags_count', text_out_name='prompt')

        modules = [
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            shuffle_tags,
        ]

        return modules


    def _preparation_modules(self, config: TrainConfig, model: WuerstchenModel):
        downscale_image = ScaleImage(in_name='image', out_name='image', factor=0.75)
        normalize_image = NormalizeImageChannels(image_in_name='image', image_out_name='image', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        encode_image = EncodeWuerstchenEffnet(in_name='image', out_name='latent_image', effnet_encoder=model.effnet_encoder, autocast_contexts=[model.autocast_context, model.effnet_encoder_autocast_context], dtype=model.effnet_encoder_train_dtype.torch_dtype())
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.75 * 0.03125) # *0.75 / 32.0
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.prior_tokenizer, max_token_length=model.prior_tokenizer.model_max_length)
        if model.model_type.is_wuerstchen_v2():
            encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name=None, add_layer_norm=True, text_encoder=model.prior_text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        elif model.model_type.is_stable_cascade():
            encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name='pooled_text_encoder_output', add_layer_norm=False, text_encoder=model.prior_text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())

        modules = [
            downscale_image, normalize_image, encode_image,
            tokenize_prompt,
        ]

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        if not config.text_encoder.train and not config.train_any_embedding():
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

        text_split_names = ['tokens', 'text_encoder_hidden_state']
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

            if not config.text_encoder.train and not config.train_any_embedding():
                modules.append(text_disk_cache)
                sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(names=sort_names, balancing_in_name='concept.balancing', balancing_strategy_in_name='concept.balancing_strategy', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], group_enabled_in_name='concept.enabled')
            modules.append(variation_sorting)

        return modules


    def _output_modules(self, config: TrainConfig, model: WuerstchenModel):
        output_names = [
            'image_path', 'latent_image',
            'tokens',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if not config.text_encoder.train and not config.train_any_embedding():
            output_names.append('text_encoder_hidden_state')

            if model.model_type.is_stable_cascade():
                output_names.append('pooled_text_encoder_output')

        sort_names = output_names + ['concept']
        output_names = output_names + [('concept.loss_weight', 'loss_weight')]

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.effnet_encoder_to(self.train_device)
            model.eval()
            torch_gc()

        mask_remove = RandomLatentMaskRemove(
            latent_mask_name='latent_mask', latent_conditioning_image_name=None,
            replace_probability=config.unmasked_probability, vae=None, possible_resolutions_in_name='possible_resolutions',
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
            before_cache_fun=before_cache_image_fun
        )
        if config.latent_caching:
            batch_sorting = AspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size)
        else:
            batch_sorting = InlineAspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=config.batch_size)

        output = OutputPipelineModule(names=output_names)

        modules = []

        if config.model_type.has_mask_input():
            modules.append(mask_remove)

        modules.append(batch_sorting)

        modules.append(output)

        return modules


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
    ):
        enumerate_input = self._enumerate_input_modules(config)
        load_input = self._load_input_modules(config, model)
        mask_augmentation = self._mask_augmentation_modules(config)
        aspect_bucketing_in = self._aspect_bucketing_in(config)
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
            train_progress
        )
