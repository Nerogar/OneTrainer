import json

from mgds.DebugDataLoaderModules import DecodeVAE, SaveImage
from mgds.DiffusersDataLoaderModules import *
from mgds.GenericDataLoaderModules import *
from mgds.MGDS import MGDS, TrainDataLoader
from mgds.TransformersDataLoaderModules import *
from transformers import CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.ModelType import ModelType


class MgdsStableDiffusionDataLoader:
    def __init__(
            self,
            args: TrainArgs,
            model: StableDiffusionModel,
    ):
        with open(args.concept_file_name, 'r') as f:
            concepts = json.load(f)

        self.ds = create_dataset(
            args.model_type, concepts, args.cache_dir, args.batch_size, device=args.train_device, dtype=args.train_dtype,
            tokenizer=model.tokenizer, vae=model.vae, image_depth_processor=model.image_depth_processor, depth_estimator=model.depth_estimator,
        )
        self.dl = TrainDataLoader(self.ds, args.batch_size)


def create_dataset(
        model_type: ModelType, concepts: list[dict], cache_dir: str, batch_size: int, device: torch.device, dtype: torch.dtype,
        tokenizer: CLIPTokenizer, vae: AutoencoderKL, image_depth_processor: DPTImageProcessor, depth_estimator: DPTForDepthEstimation
):
    size = 768

    input_modules = [
        CollectPaths(concept_in_name='concept', path_in_name='path', name_in_name='name', path_out_name='image_path', concept_out_name='concept', extensions=['.png', '.jpg'], include_postfix=None,
                     exclude_postfix=['-masklabel']),
        ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png'),
        ModifyPath(in_name='image_path', out_name='prompt_path', postfix='', extension='.txt'),
        LoadImage(path_in_name='image_path', image_out_name='image', range_min=-1.0, range_max=1.0),
        LoadImage(path_in_name='mask_path', image_out_name='mask', range_min=0, range_max=1, channels=1),
        GenerateDepth(path_in_name='image_path', image_out_name='depth', image_depth_processor=image_depth_processor, depth_estimator=depth_estimator) if model_type.has_depth_input() else None,
        #RandomCircularMaskShrink(mask_name='mask', shrink_probability=1.0, shrink_factor_min=0.2, shrink_factor_max=1.0),
        RandomMaskRotateCrop(mask_name='mask', additional_names=['image', 'depth'], min_size=size, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20) if model_type.has_depth_input() else
        RandomMaskRotateCrop(mask_name='mask', additional_names=['image'], min_size=size, min_padding_percent=10, max_padding_percent=30, max_rotate_angle=20),
        CalcAspect(image_in_name='image', resolution_out_name='original_resolution'),
        AspectBucketing(batch_size=batch_size, target_resolution=size, resolution_in_name='original_resolution', scale_resolution_out_name='scale_resolution', crop_resolution_out_name='crop_resolution',
                        possible_resolutions_out_name='possible_resolutions'),
        ScaleCropImage(image_in_name='image', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', image_out_name='image'),
        ScaleCropImage(image_in_name='mask', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', image_out_name='mask'),
        ScaleCropImage(image_in_name='depth', scale_resolution_in_name='scale_resolution', crop_resolution_in_name='crop_resolution', image_out_name='depth') if model_type.has_depth_input() else None,
        LoadText(path_in_name='prompt_path', text_out_name='prompt'),
        GenerateMaskedConditioningImage(image_in_name='image', mask_in_name='mask', image_out_name='conditioning_image'),
        RandomFlip(names=['image', 'mask', 'depth', 'conditioning_image']) if model_type.has_depth_input() else
        RandomFlip(names=['image', 'mask', 'conditioning_image']),
        EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=vae),
        Downscale(in_name='mask', out_name='latent_mask'),
        EncodeVAE(in_name='conditioning_image', out_name='latent_conditioning_image_distribution', vae=vae),
        Downscale(in_name='depth', out_name='latent_depth') if model_type.has_depth_input() else None,
        Tokenize(in_name='prompt', out_name='tokens', tokenizer=tokenizer),
        DiskCache(cache_dir=cache_dir, split_names=['latent_image_distribution', 'latent_mask', 'latent_conditioning_image_distribution', 'latent_depth', 'tokens'], aggregate_names=['crop_resolution']) if model_type.has_depth_input() else
        DiskCache(cache_dir=cache_dir, split_names=['latent_image_distribution', 'latent_mask', 'latent_conditioning_image_distribution', 'tokens'], aggregate_names=['crop_resolution']),
        SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean'),
        SampleVAEDistribution(in_name='latent_conditioning_image_distribution', out_name='latent_conditioning_image', mode='mean'),
        RandomLatentMaskRemove(latent_mask_name='latent_mask', latent_conditioning_image_name='latent_conditioning_image', replace_probability=0.0, vae=vae, possible_resolutions_in_name='possible_resolutions')
    ]

    debug_modules = [
        DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=vae),
        DecodeVAE(in_name='latent_conditioning_image', out_name='decoded_conditioning_image', vae=vae),
        SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=cache_dir+'/debug', in_range_min=-1, in_range_max=1),
        SaveImage(image_in_name='mask', original_path_in_name='image_path', path=cache_dir+'/debug', in_range_min=0, in_range_max=1),
        SaveImage(image_in_name='decoded_conditioning_image', original_path_in_name='image_path', path=cache_dir+'/debug', in_range_min=-1, in_range_max=1),
        # SaveImage(image_in_name='depth', original_path_in_name='image_path', path=cache_dir+'/debug', in_range_min=-1, in_range_max=1),
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=cache_dir+'/debug', in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='latent_depth', original_path_in_name='image_path', path=cache_dir+'/debug', in_range_min=-1, in_range_max=1),
    ]

    output_modules = [
        AspectBatchSorting(resolution_in_name='crop_resolution', names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'latent_depth', 'tokens'], batch_size=batch_size,
                           sort_resolutions_for_each_epoch=True) if model_type.has_depth_input() else
        AspectBatchSorting(resolution_in_name='crop_resolution', names=['latent_image', 'latent_conditioning_image', 'latent_mask', 'tokens'], batch_size=batch_size, sort_resolutions_for_each_epoch=True),
    ]

    ds = MGDS(
        device, dtype,
        concepts,
        [
            input_modules,
            # debug_modules,
            output_modules
        ]
    )

    return ds
