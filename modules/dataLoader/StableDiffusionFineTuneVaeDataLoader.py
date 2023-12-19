import json
import os

import torch
from mgds.MGDS import TrainDataLoader, MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RamCache import RamCache
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util import path_util
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.torch_util import torch_gc


class StableDiffusionFineTuneVaeDataLoader(BaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionFineTuneVaeDataLoader, self).__init__(
            train_device,
            temp_device,
        )

        with open(args.concept_file_name, 'r') as f:
            concepts = json.load(f)

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
            model: StableDiffusionModel,
            train_device: torch.device,
            temp_device: torch.device,
            args: TrainArgs,
    ):
        model.to(self.temp_device)

        model.vae_to(train_device)

        model.eval()
        torch_gc()

    def __enumerate_input_modules(self, args: TrainArgs) -> list:
        supported_extensions = path_util.supported_image_extensions()

        collect_paths = CollectPaths(
            concept_in_name='concept', path_in_name='path', name_in_name='name', path_out_name='image_path',
            concept_out_name='concept', extensions=supported_extensions, include_postfix=None,
            exclude_postfix=['-masklabel'], include_subdirectories_in_name='concept.include_subdirectories'
        )

        mask_path = ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png')

        modules = [collect_paths]

        if args.masked_training:
            modules.append(mask_path)

        return modules


    def __load_input_modules(self, args: TrainArgs) -> list:
        load_image = LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0)
        load_mask = LoadImage(path_in_name='mask_path', image_out_name='latent_mask', range_min=0, range_max=1, channels=1)

        modules = [load_image]

        if args.masked_training:
            modules.append(load_mask)

        return modules


    def __mask_augmentation_modules(self, args: TrainArgs) -> list:
        inputs = ['image']

        lowest_resolution = min(int(res.strip()) for res in args.resolution.split(','))

        random_mask_rotate_crop = RandomMaskRotateCrop(mask_name='latent_mask', additional_names=inputs, min_size=lowest_resolution,
                                                       min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20,
                                                       enabled_in_name='settings.enable_random_circular_mask_shrink')

        modules = []

        if args.masked_training and args.random_rotate_and_crop:
            modules.append(random_mask_rotate_crop)

        return modules


    def __aspect_bucketing_in(self, args: TrainArgs):
        calc_aspect = CalcAspect(image_in_name='image', resolution_out_name='original_resolution')
        aspect_bucketing = AspectBucketing(
            target_resolution=[int(res.strip()) for res in args.resolution.split(',')],
            quantization=8,
            resolution_in_name='original_resolution',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        single_aspect_calculation = SingleAspectCalculation(
            target_resolution=[int(res.strip()) for res in args.resolution.split(',')],
            resolution_in_name='original_resolution',
            scale_resolution_out_name='scale_resolution',
            crop_resolution_out_name='crop_resolution',
            possible_resolutions_out_name='possible_resolutions'
        )

        modules = []

        modules.append(calc_aspect)
        if args.aspect_ratio_bucketing:
            modules.append(aspect_bucketing)
        else:
            modules.append(single_aspect_calculation)

        return modules


    def __crop_modules(self, args: TrainArgs):
        scale_crop_image = ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution',
                                          crop_resolution_in_name='crop_resolution',
                                          enable_crop_jitter_in_name='concept.enable_crop_jitter',
                                          image_out_name='image', crop_offset_out_name='crop_offset')
        scale_crop_mask = ScaleCropImage(image_in_name='latent_mask', scale_resolution_in_name='scale_resolution',
                                         crop_resolution_in_name='crop_resolution',
                                         enable_crop_jitter_in_name='concept.enable_crop_jitter',
                                         image_out_name='latent_mask', crop_offset_out_name='crop_offset')

        modules = [scale_crop_image]

        if args.masked_training:
            modules.append(scale_crop_mask)

        return modules


    def __augmentation_modules(self, args: TrainArgs):
        inputs = ['image']

        if args.masked_training:
            inputs.append('latent_mask')

        random_flip = RandomFlip(names=inputs, enabled_in_name='concept.enable_random_flip')

        modules = [random_flip]

        return modules


    def __preparation_modules(self, args: TrainArgs, model: StableDiffusionModel):
        image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae)

        modules = [image]

        return modules


    def __cache_modules(self, args: TrainArgs, model: StableDiffusionModel):
        split_names = ['image', 'latent_image_distribution']

        if args.masked_training:
            split_names.append('latent_mask')

        aggregate_names = ['crop_resolution', 'image_path']

        def before_cache_fun():
            self._setup_cache_device(model, self.train_device, self.temp_device, args)

        disk_cache = DiskCache(cache_dir=args.cache_dir, split_names=split_names, aggregate_names=aggregate_names, variations_in_name='concept.image_variations', repeats_in_name='concept.repeats', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_fun)
        ram_cache = RamCache(cache_names=split_names + aggregate_names, repeats_in_name='concept.repeats', variations_group_in_name=['concept.path', 'concept.seed', 'concept.include_subdirectories', 'concept.image'], group_enabled_in_name='concept.enabled', before_cache_fun=before_cache_fun)

        modules = []

        if args.latent_caching:
            modules.append(disk_cache)
        else:
            modules.append(ram_cache)

        return modules


    def __output_modules(self, args: TrainArgs):
        output_names = ['image', 'latent_image', 'image_path']

        if args.masked_training:
            output_names.append('latent_mask')

        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        batch_sorting = AspectBatchSorting(resolution_in_name='crop_resolution', names=output_names,
                                           batch_size=args.batch_size)
        output = OutputPipelineModule(names=output_names)

        modules = [image_sample]

        if args.aspect_ratio_bucketing:
            modules.append(batch_sorting)

        modules.append(output)

        return modules


    def create_dataset(
            self,
            args: TrainArgs,
            model: StableDiffusionModel,
            concepts: list[dict],
            train_progress: TrainProgress,
    ):
        enumerate_input = self.__enumerate_input_modules(args)
        load_input = self.__load_input_modules(args)
        mask_augmentation = self.__mask_augmentation_modules(args)
        aspect_bucketing_in = self.__aspect_bucketing_in(args)
        crop_modules = self.__crop_modules(args)
        augmentation_modules = self.__augmentation_modules(args)
        preparation_modules = self.__preparation_modules(args, model)
        cache_modules = self.__cache_modules(args, model)
        output_modules = self.__output_modules(args)

        debug_dir = os.path.join(args.debug_dir, "dataloader")
        debug_modules = [
            DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae),
            SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1,
                      in_range_max=1),
            SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0,
                      in_range_max=1),
            # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        ]

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
            ],
            train_progress
        )
