import json
import os

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
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.NormalizeImageChannels import NormalizeImageChannels
from mgds.pipelineModules.RamCache import RamCache
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
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
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.params.ConceptParams import ConceptParams
from modules.util.torch_util import torch_gc


class WuerstchenBaseDataLoader(BaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
            model: WuerstchenModel,
            train_progress: TrainProgress,
    ):
        super(WuerstchenBaseDataLoader, self).__init__(
            train_device,
            temp_device,
        )

        with open(args.concept_file_name, 'r') as f:
            concepts = json.load(f)
            for i in range(len(concepts)):
                concepts[i] = ConceptParams.default_values().from_dict(concepts[i]).to_dict()

        self.__ds = self.create_dataset(
            args=args,
            model=model,
            concepts=concepts,
            train_progress=train_progress,
        )
        self.__dl = TrainDataLoader(self.__ds, args.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    def _setup_cache_device(
            self,
            model: WuerstchenModel,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
    ):
        model.to(self.temp_device)

        model.effnet_encoder_to(train_device)
        if not args.train_text_encoder and args.training_method != TrainingMethod.EMBEDDING:
            model.prior_text_encoder_to(train_device)

        model.eval()
        torch_gc()

    def _enumerate_input_modules(self, args: TrainArgs) -> list:
        supported_extensions = path_util.supported_image_extensions()

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', path_out_name='image_path', concept_out_name='concept',
            extensions=supported_extensions, include_postfix=None, exclude_postfix=['-masklabel'], include_subdirectories_in_name='concept.include_subdirectories'
        )

        mask_path = ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png')
        sample_prompt_path = ModifyPath(in_name='image_path', out_name='sample_prompt_path', postfix='', extension='.txt')

        modules = [collect_paths, sample_prompt_path]

        if args.masked_training:
            modules.append(mask_path)

        return modules


    def _load_input_modules(self, args: TrainArgs, model: WuerstchenModel) -> list:
        load_image = LoadImage(path_in_name='image_path', image_out_name='image', range_min=0, range_max=1)

        generate_mask = GenerateImageLike(image_in_name='image', image_out_name='mask', color=255, range_min=0, range_max=1, channels=1)
        load_mask = LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1)

        load_sample_prompts = LoadMultipleTexts(path_in_name='sample_prompt_path', texts_out_name='sample_prompts')
        load_concept_prompts = LoadMultipleTexts(path_in_name='concept.text.prompt_path', texts_out_name='concept_prompts')
        filename_prompt = GetFilename(path_in_name='image_path', filename_out_name='filename_prompt', include_extension=False)
        select_prompt_input = SelectInput(setting_name='concept.text.prompt_source', out_name='prompts', setting_to_in_name_map={
            'sample': 'sample_prompts',
            'concept': 'concept_prompts',
            'filename': 'filename_prompt',
        }, default_in_name='sample_prompts')
        select_random_text = SelectRandomText(texts_in_name='prompts', text_out_name='prompt')

        modules = [load_image, load_sample_prompts, load_concept_prompts, filename_prompt, select_prompt_input, select_random_text]

        if args.masked_training:
            modules.append(generate_mask)
            modules.append(load_mask)
        elif args.model_type.has_mask_input():
            modules.append(generate_mask)

        return modules


    def _mask_augmentation_modules(self, args: TrainArgs) -> list:
        inputs = ['image']

        lowest_resolution = min(int(res.strip()) for res in args.resolution.split(','))
        circular_mask_shrink = RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0, enabled_in_name='settings.enable_random_circular_mask_shrink')
        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='mask', additional_names=inputs, min_size=lowest_resolution, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20, enabled_in_name='settings.enable_random_mask_rotate_crop')

        modules = []

        if args.masked_training or args.model_type.has_mask_input():
            modules.append(circular_mask_shrink)

        if args.masked_training or args.model_type.has_mask_input():
            modules.append(random_mask_rotate_crop)

        return modules


    def _aspect_bucketing_in(self, args: TrainArgs):
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

        if args.aspect_ratio_bucketing:
            modules.append(aspect_bucketing)
        else:
            modules.append(single_aspect_calculation)

        return modules


    def _crop_modules(self, args: TrainArgs):
        scale_crop_image = ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='image', crop_offset_out_name='crop_offset')
        scale_crop_mask = ScaleCropImage(image_in_name='mask', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', enable_crop_jitter_in_name='concept.enable_crop_jitter', image_out_name='mask', crop_offset_out_name='crop_offset')

        modules = [scale_crop_image]

        if args.masked_training or args.model_type.has_mask_input():
            modules.append(scale_crop_mask)

        return modules


    def _augmentation_modules(self, args: TrainArgs):
        inputs = ['image']

        if args.masked_training or args.model_type.has_mask_input():
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


    def _preparation_modules(self, args: TrainArgs, model: WuerstchenModel):
        downscale_image = ScaleImage(in_name='image', out_name='image', factor=0.75)
        normalize_image = NormalizeImageChannels(image_in_name='image', image_out_name='image', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        encode_image = EncodeWuerstchenEffnet(in_name='image', out_name='latent_image', effnet_encoder=model.effnet_encoder, override_allow_mixed_precision=False)
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.75)
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.prior_tokenizer, max_token_length=model.prior_tokenizer.model_max_length)
        encode_prompt = EncodeClipText(in_name='tokens', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name=None, add_layer_norm=True, text_encoder=model.prior_text_encoder, hidden_state_output_index=-1)

        modules = [
            downscale_image, normalize_image, encode_image,
            tokenize_prompt,
        ]

        if args.masked_training or args.model_type.has_mask_input():
            modules.append(downscale_mask)

        if not args.train_text_encoder and args.training_method != TrainingMethod.EMBEDDING:
            modules.append(encode_prompt)

        return modules


    def _cache_modules(self, args: TrainArgs, model: WuerstchenModel):
        image_split_names = [
            'latent_image',
            'original_resolution', 'crop_offset',
        ]

        if args.masked_training or args.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = ['tokens', 'text_encoder_hidden_state']

        sort_names = text_split_names + [
            'prompt', 'concept'
        ]

        image_cache_dir = os.path.join(args.cache_dir, "image")
        text_cache_dir = os.path.join(args.cache_dir, "text")

        def before_cache_fun():
            self._setup_cache_device(model, self.train_device, self.temp_device, args)

        image_disk_cache = DiskCache(cache_dir=image_cache_dir, split_names=image_split_names, aggregate_names=image_aggregate_names, variations_in_name='concept.image_variations', repeats_in_name='concept.repeats', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_fun)
        image_ram_cache = RamCache(cache_names=image_split_names + image_aggregate_names, repeats_in_name='concept.repeats', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_fun)

        text_disk_cache = DiskCache(cache_dir=text_cache_dir, split_names=text_split_names, aggregate_names=[], variations_in_name='concept.text_variations', repeats_in_name='concept.repeats', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_fun)

        modules = []

        if args.latent_caching:
            modules.append(image_disk_cache)
        else:
            modules.append(image_ram_cache)

        if not args.train_text_encoder and args.latent_caching and args.training_method != TrainingMethod.EMBEDDING:
            modules.append(text_disk_cache)
            sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(names=sort_names, repeats_in_name='concept.repeats', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.text'], group_enabled_in_name='concept.enabled')
            modules.append(variation_sorting)

        return modules


    def _output_modules(self, args: TrainArgs, model: WuerstchenModel):
        output_names = [
            'image_path', 'latent_image',
            'tokens',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if args.masked_training or args.model_type.has_mask_input():
            output_names.append('latent_mask')

        if not args.train_text_encoder and args.training_method != TrainingMethod.EMBEDDING:
            output_names.append('text_encoder_hidden_state')

        sort_names = output_names + ['concept']
        output_names = output_names + [('concept.loss_weight', 'loss_weight')]

        mask_remove = RandomLatentMaskRemove(
            latent_mask_name='latent_mask', latent_conditioning_image_name=None,
            replace_probability=args.unmasked_probability, vae=None, possible_resolutions_in_name='possible_resolutions'
        )
        batch_sorting = AspectBatchSorting(resolution_in_name='crop_resolution', names=sort_names, batch_size=args.batch_size)
        output = OutputPipelineModule(names=output_names)

        modules = []

        if args.model_type.has_mask_input():
            modules.append(mask_remove)

        modules.append(batch_sorting)

        modules.append(output)

        return modules


    def _debug_modules(self, args: TrainArgs, model: WuerstchenModel):
        debug_dir = os.path.join(args.debug_dir, "dataloader")

        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=1.0/0.75)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.prior_tokenizer)
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir)

        # These modules don't really work, since they are inserted after a sorting operation that does not include this data
        # SaveImage(image_in_name='mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),

        modules = []

        if args.masked_training or args.model_type.has_mask_input():
            modules.append(upscale_mask)
            modules.append(save_mask)

        modules.append(decode_prompt)
        modules.append(save_prompt)

        return modules


    def create_dataset(
            self,
            args: TrainArgs,
            model: WuerstchenModel,
            concepts: list[dict],
            train_progress: TrainProgress,
    ):
        enumerate_input = self._enumerate_input_modules(args)
        load_input = self._load_input_modules(args, model)
        mask_augmentation = self._mask_augmentation_modules(args)
        aspect_bucketing_in = self._aspect_bucketing_in(args)
        crop_modules = self._crop_modules(args)
        augmentation_modules = self._augmentation_modules(args)
        preparation_modules = self._preparation_modules(args, model)
        cache_modules = self._cache_modules(args, model)
        output_modules = self._output_modules(args, model)

        debug_modules = self._debug_modules(args, model)

        return self._create_mgds(
            args,
            concepts,
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

                debug_modules if args.debug_mode else None,
                # inserted before output_modules, which contains a sorting operation
            ],
            train_progress
        )
