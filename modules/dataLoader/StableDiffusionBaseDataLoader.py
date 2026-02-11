import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.BaseModel import BaseModel
from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionSetup import BaseStableDiffusionSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeClipText import EncodeClipText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


class StableDiffusionBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: StableDiffusionModel):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        rescale_conditioning_image = RescaleImageChannels(image_in_name='conditioning_image', image_out_name='conditioning_image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.125)
        add_embeddings_to_prompt = MapData(in_name='prompt', out_name='prompt', map_fn=model.add_text_encoder_embeddings_to_prompt)
        encode_conditioning_image = EncodeVAE(in_name='conditioning_image', out_name='latent_conditioning_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        conditioning_image_sample = SampleVAEDistribution(in_name='latent_conditioning_image_distribution', out_name='latent_conditioning_image', mode='mean')
        downscale_depth = ScaleImage(in_name='depth', out_name='latent_depth', factor=0.125)
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.tokenizer, max_token_length=model.tokenizer.model_max_length)
        encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name=None, hidden_state_out_name='text_encoder_hidden_state', pooled_out_name=None, add_layer_norm=True,
                                       text_encoder=model.text_encoder, hidden_state_output_index=-(1 + config.text_encoder_layer_skip), autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())

        modules = [rescale_image, encode_image, image_sample]

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        if config.model_type.has_conditioning_image_input():
            modules += [rescale_conditioning_image, encode_conditioning_image, conditioning_image_sample]

        if config.model_type.has_depth_input():
            modules.append(downscale_depth)

        modules += [add_embeddings_to_prompt, tokenize_prompt]
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)

        return modules

    def _cache_modules(self, config: TrainConfig, model: StableDiffusionModel, model_setup: BaseStableDiffusionSetup):
        image_split_names = ['latent_image']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            image_split_names.append('latent_conditioning_image')

        if config.model_type.has_depth_input():
            image_split_names.append('latent_depth')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = ['tokens', 'text_encoder_hidden_state']

        sort_names = text_split_names + image_aggregate_names + image_split_names + [
            'prompt', 'concept'
        ]

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: StableDiffusionModel, model_setup: BaseStableDiffusionSetup):
        output_names = [
            'image_path', 'latent_image',
            'prompt',
            'tokens',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        if config.model_type.has_conditioning_image_input():
            output_names.append('latent_conditioning_image')

        if config.model_type.has_depth_input():
            output_names.append('latent_depth')

        if not config.train_text_encoder_or_embedding():
            output_names.append('text_encoder_hidden_state')

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=True,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: StableDiffusionModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        decode_conditioning_image = DecodeVAE(in_name='latent_conditioning_image', out_name='decoded_conditioning_image', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=8)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.tokenizer)
        save_image = SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        save_conditioning_image = SaveImage(image_in_name='decoded_conditioning_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        # SaveImage(image_in_name='latent_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        # SaveImage(image_in_name='latent_depth', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir, before_save_fun=before_save_fun)

        # These modules don't really work, since they are inserted after a sorting operation that does not include this data
        # SaveImage(image_in_name='mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1),
        # SaveImage(image_in_name='depth', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),
        # SaveImage(image_in_name='image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1),

        modules = [decode_image, save_image]

        if config.model_type.has_conditioning_image_input():
            modules += [decode_conditioning_image, save_conditioning_image]

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
            aspect_bucketing_quantization=8,
        )

factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_15)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_15_INPAINTING)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_20)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_20_BASE)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_20_INPAINTING)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_20_DEPTH)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_21)
factory.register(BaseDataLoader, StableDiffusionBaseDataLoader, ModelType.STABLE_DIFFUSION_21_BASE)
