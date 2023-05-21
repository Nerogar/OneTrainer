from mgds.GenericDataLoaderModules import *

from modules.dataLoader.MgdsStableDiffusionBaseDataLoader import MgdsStablDiffusionBaseDataLoader
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs


class MgdsStableDiffusionFineTuneDataLoader(MgdsStablDiffusionBaseDataLoader):
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(MgdsStableDiffusionFineTuneDataLoader, self).__init__(args, model, train_progress)



    def _cache_modules(self, args: TrainArgs):
        split_names = ['latent_image_distribution', 'tokens']

        if args.masked_training or args.model_type.has_mask_input():
            split_names.append('latent_mask')

        if args.model_type.has_conditioning_image_input():
            split_names.append('latent_conditioning_image_distribution')

        if args.model_type.has_depth_input():
            split_names.append('latent_depth')

        aggregate_names = ['crop_resolution', 'image_path']

        disk_cache = DiskCache(cache_dir=args.cache_dir, split_names=split_names, aggregate_names=aggregate_names, cached_epochs=args.latent_caching_epochs)

        modules = []

        if args.latent_caching:
            modules.append(disk_cache)

        return modules


