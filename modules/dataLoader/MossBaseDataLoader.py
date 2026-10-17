import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.BaseModel import BaseModel
from modules.model.MossModel import (
    PROMPT_MAX_LENGTH,
    MossModel,
)
from modules.modelSetup.BaseMossSetup import BaseMossSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.PadMaskedTokens import PadMaskedTokens
from mgds.pipelineModules.PruneMaskedTokens import PruneMaskedTokens
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


def _moss_chat_messages(text: str) -> list[dict]:
    return [{'role': 'user', 'content': text}]


class MossTokenize(Tokenize):
    """Tokenize without padding to max_length.

    The Qwen text encoder is causal, so the hidden states of the real tokens are
    identical with or without trailing padding. mgds' Tokenize always pads to
    max_length, which makes the encoder burn ~4000 pad tokens per caption and
    is what makes text caching painfully slow. This subclass tokenizes
    unpadded ('longest'); format_text/suffix_text/expand_mask are not used by
    the MOSS pipeline, so they are intentionally not re-implemented here.
    """

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.in_name, index)

        with self.tokenizer_lock:
            if self.apply_chat_template is not None:
                messages = self.apply_chat_template(text)
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                )

            tokenizer_output = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_token_length,
                return_tensors="pt",
            )

        tokens = tokenizer_output.input_ids.to(self.pipeline.device)
        mask = tokenizer_output.attention_mask.to(self.pipeline.device)

        return {
            self.tokens_out_name: tokens.squeeze(dim=0),
            self.mask_out_name: mask.squeeze(dim=0),
        }


@factory.register(BaseDataLoader, ModelType.MOSS)
class MossBaseDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: MossModel):
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=1 / 16)
        # MOSS prompts are wrapped in the chat template (user turn, generation prompt, thinking enabled)
        # the Qwen encoder is causal, so real-token hidden states are identical with or
        # without padding -- tokenize unpadded ('longest') to avoid encoding ~4000 pad
        # tokens per caption, which is what makes text caching painfully slow.
        tokenize_prompt = MossTokenize(in_name='prompt', tokens_out_name='tokens', mask_out_name='tokens_mask', tokenizer=model.tokenizer, max_token_length=PROMPT_MAX_LENGTH,
                                        apply_chat_template=_moss_chat_messages, apply_chat_template_kwargs={'add_generation_prompt': True, 'enable_thinking': True})
        # single last hidden state, no crop, encoder default position ids
        encode_prompt = EncodeQwenText(tokens_name='tokens', tokens_attention_mask_in_name='tokens_mask', hidden_state_out_name='text_encoder_hidden_state', tokens_attention_mask_out_name='tokens_mask',
                                       text_encoder=model.text_encoder, hidden_state_output_index=-1, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
                                       cumsum_position_ids=False)
        prune_masked_tokens = PruneMaskedTokens(tokens_name='tokens', tokens_mask_name='tokens_mask', hidden_state_name='text_encoder_hidden_state')

        modules = [rescale_image, encode_image, image_sample]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        modules.append(tokenize_prompt)

        if not config.train_text_encoder_or_embedding():
            modules.append(encode_prompt)

        if config.latent_caching and not config.train_text_encoder_or_embedding():
            modules.append(prune_masked_tokens)

        return modules

    def _cache_modules(self, config: TrainConfig, model: MossModel, model_setup: BaseMossSetup):
        image_split_names = ['latent_image', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']

        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens', 'tokens_mask', 'text_encoder_hidden_state',
            'concept'
        ]

        if not config.train_text_encoder_or_embedding():
            text_split_names += ['tokens', 'tokens_mask', 'text_encoder_hidden_state']

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=not config.train_text_encoder_or_embedding(),
        )

    def _output_modules(self, config: TrainConfig, model: MossModel, model_setup: BaseMossSetup):
        pad_masked_tokens = PadMaskedTokens(tokens_name='tokens', tokens_mask_name='tokens_mask', hidden_state_name='text_encoder_hidden_state', max_length=PROMPT_MAX_LENGTH)

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

        output_module_list = self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

        if config.latent_caching and not config.train_text_encoder_or_embedding():
            output_module_list = [pad_masked_tokens] + output_module_list

        return output_module_list

    def _debug_modules(self, config: TrainConfig, model: MossModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.tokenizer)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir)

        # NOTE: the MOSS VAE decodes 128-channel latents via a batch-norm denorm + 2x2 channel unfold,
        # which the generic DecodeVAE module (a plain vae.decode) cannot perform, so image decode is
        # omitted from the debug pipeline (mirrors Krea 2, where image saving is also disabled).
        modules = [decode_prompt, save_prompt]

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
            allow_video_files=False,
            vae_frame_dim=False,
        )
