import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.wuerstchen.EncodeWuerstchenEffnet import EncodeWuerstchenEffnet
from modules.model.BaseModel import BaseModel
from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseWuerstchenSetup import BaseWuerstchenSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.EncodeClipText import EncodeClipText
from mgds.pipelineModules.MapData import MapData
from mgds.pipelineModules.NormalizeImageChannels import NormalizeImageChannels
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


class WuerstchenBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: WuerstchenModel):
        downscale_image = ScaleImage(in_name='image', out_name='image', factor=0.75)
        normalize_image = NormalizeImageChannels(image_in_name='image', image_out_name='image', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        encode_image = EncodeWuerstchenEffnet(in_name='image', out_name='latent_image', effnet_encoder=model.effnet_encoder, autocast_contexts=[model.autocast_context, model.effnet_encoder_autocast_context], dtype=model.effnet_encoder_train_dtype.torch_dtype())
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.75 * 0.03125) # *0.75 / 32.0
        add_embeddings_to_prompt = MapData(in_name='prompt', out_name='prompt', map_fn=model.add_prior_text_encoder_embeddings_to_prompt)
        tokenize_prompt = Tokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.prior_tokenizer, max_token_length=model.prior_tokenizer.model_max_length)
        if model.model_type.is_wuerstchen_v2():
            encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name=None, add_layer_norm=True,
                                           text_encoder=model.prior_text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        elif model.model_type.is_stable_cascade():
            encode_prompt = EncodeClipText(in_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', pooled_out_name='pooled_text_encoder_output', add_layer_norm=False,
                                           text_encoder=model.prior_text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())

        modules = [downscale_image, normalize_image, encode_image]

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        modules += [add_embeddings_to_prompt, tokenize_prompt]
        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)

        return modules

    def _cache_modules(self, config: TrainConfig, model: WuerstchenModel, model_setup: BaseWuerstchenSetup):
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

        def before_cache_image_fun():
            model.to(self.temp_device)
            model.effnet_encoder_to(self.train_device)
            model.eval()
            torch_gc()

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
            before_cache_image_fun=before_cache_image_fun
        )

    def _output_modules(self, config: TrainConfig, model: WuerstchenModel, model_setup: BaseWuerstchenSetup):
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
            model, model_setup,
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
            aspect_bucketing_quantization=128,
            supports_inpainting=False,
        )

factory.register(BaseDataLoader, WuerstchenBaseDataLoader, ModelType.WUERSTCHEN_2)
factory.register(BaseDataLoader, WuerstchenBaseDataLoader, ModelType.STABLE_CASCADE_1)
