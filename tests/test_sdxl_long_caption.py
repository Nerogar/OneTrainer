import os
import unittest

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import sdxl_text_cache_dir
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.model.util.clip_chunk import (
    CLIP_CHUNK_TOKENS,
    CLIP_CONTENT_TOKENS,
    encode_clip_chunks,
    sdxl_chunk_attention_mask,
    sdxl_chunk_aware_unet_prediction,
    tokenize_clip_chunks,
    validate_sdxl_caption_token_limit,
)
from modules.model.util.clip_util import encode_clip
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType

import torch

from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def _small_clip_encoders(
        tokenizers: tuple[CLIPTokenizer, CLIPTokenizer],
) -> tuple[CLIPTextModel, CLIPTextModelWithProjection]:
    def config(tokenizer: CLIPTokenizer, width: int) -> CLIPTextConfig:
        return CLIPTextConfig(
            vocab_size=len(tokenizer),
            hidden_size=width,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=CLIP_CHUNK_TOKENS,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            projection_dim=width,
        )

    return CLIPTextModel(config(tokenizers[0], 16)), CLIPTextModelWithProjection(config(tokenizers[1], 24))


def _make_xl_model(tokenizers: tuple[CLIPTokenizer, CLIPTokenizer]) -> StableDiffusionXLModel:
    torch.manual_seed(321)
    model = StableDiffusionXLModel(ModelType.STABLE_DIFFUSION_XL_10_BASE)
    model.tokenizer_1, model.tokenizer_2 = tokenizers
    model.text_encoder_1, model.text_encoder_2 = _small_clip_encoders(tokenizers)
    model.text_encoder_1.eval()
    model.text_encoder_2.eval()
    return model


class SDXLLongCaptionUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = {"!</w>": 0, "a</w>": 1, "b</w>": 2, "<|startoftext|>": 3, "<|endoftext|>": 4}
        cls.tokenizers = (
            CLIPTokenizer(vocab=vocab, model_max_length=CLIP_CHUNK_TOKENS),
            CLIPTokenizer(vocab=vocab, pad_token="!</w>", model_max_length=CLIP_CHUNK_TOKENS),
        )

    def test_capacity_validation_and_cache_namespaces(self):
        for token_limit in (75, 150, 225):
            self.assertEqual(validate_sdxl_caption_token_limit(token_limit), token_limit)
        with self.assertRaisesRegex(ValueError, "75, 150, or 225"):
            validate_sdxl_caption_token_limit(100)
        with self.assertRaisesRegex(ValueError, "75, 150, or 225"):
            validate_sdxl_caption_token_limit(150.0)

        config = TrainConfig.default_values()
        config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
        config.cache_dir = "cache"
        config.sdxl_max_caption_tokens = 75
        self.assertEqual(sdxl_text_cache_dir(config), os.path.join("cache", "text"))
        config.sdxl_max_caption_tokens = 150
        self.assertEqual(sdxl_text_cache_dir(config), os.path.join("cache", "text", "sdxl-150"))
        config.sdxl_max_caption_tokens = 225
        self.assertEqual(sdxl_text_cache_dir(config), os.path.join("cache", "text", "sdxl-225"))

    def test_config_roundtrip_and_old_defaults(self):
        config = TrainConfig.default_values()
        config.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
        config.sdxl_max_caption_tokens = 225
        restored = TrainConfig.default_values().from_dict(config.to_dict())
        self.assertEqual(restored.sdxl_max_caption_tokens, 225)

        old_data = config.to_dict()
        old_data.pop("sdxl_max_caption_tokens")
        restored_old = TrainConfig.default_values().from_dict(old_data)
        self.assertEqual(restored_old.sdxl_max_caption_tokens, 75)

        sample = SampleConfig.default_values(ModelType.STABLE_DIFFUSION_XL_10_BASE)
        sample.from_train_config(config)
        self.assertEqual(sample.sdxl_max_caption_tokens, 225)

        non_sdxl = TrainConfig.default_values()
        non_sdxl.model_type = ModelType.STABLE_DIFFUSION_15
        non_sdxl.cache_dir = "cache"
        non_sdxl.sdxl_max_caption_tokens = 225
        self.assertEqual(sdxl_text_cache_dir(non_sdxl), os.path.join("cache", "text"))

    def test_tokenization_boundaries_for_both_sdxl_tokenizers(self):
        token_counts = (1, 74, 75, 76, 150, 151, 225, 226)
        for tokenizer in self.tokenizers:
            for token_limit in (75, 150, 225):
                chunk_count = token_limit // CLIP_CONTENT_TOKENS
                for content_count in token_counts:
                    text = "a " * content_count
                    expected_content = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=token_limit,
                    ).input_ids
                    tokens, mask = tokenize_clip_chunks(tokenizer, text, token_limit)

                    self.assertEqual(tokens.shape, (chunk_count * CLIP_CHUNK_TOKENS,))
                    self.assertEqual(mask.shape, tokens.shape)
                    chunks = tokens.reshape(chunk_count, CLIP_CHUNK_TOKENS)
                    masks = mask.reshape(chunk_count, CLIP_CHUNK_TOKENS)
                    reconstructed_content = []
                    for chunk_index, (chunk, chunk_mask) in enumerate(zip(chunks, masks, strict=True)):
                        consumed = min(max(content_count - chunk_index * CLIP_CONTENT_TOKENS, 0), CLIP_CONTENT_TOKENS)
                        self.assertEqual(chunk[0].item(), tokenizer.bos_token_id)
                        self.assertEqual(chunk[consumed + 1].item(), tokenizer.eos_token_id)
                        self.assertTrue(torch.all(chunk_mask[:consumed + 2] == 1))
                        self.assertTrue(torch.all(chunk_mask[consumed + 2:] == 0))
                        self.assertTrue(torch.all(chunk[consumed + 2:] == tokenizer.pad_token_id))
                        reconstructed_content.extend(chunk[1:consumed + 1].tolist())

                    self.assertEqual(reconstructed_content, expected_content[:token_limit])

                    if token_limit == 75:
                        legacy = tokenizer(
                            text,
                            padding="max_length",
                            truncation=True,
                            max_length=77,
                            return_tensors="pt",
                        )
                        self.assertTrue(torch.equal(tokens, legacy.input_ids[0]))
                        self.assertTrue(torch.equal(mask, legacy.attention_mask[0]))

    def test_fixed_capacity_batch_and_225_prompt_encoding(self):
        tokenizer_1, _ = self.tokenizers
        prompts = [
            "a " * 1,
            "a " * 226,
        ]
        batch = torch.stack([tokenize_clip_chunks(tokenizer_1, prompt, 225)[0] for prompt in prompts])
        self.assertEqual(batch.shape, (2, 231))

        model = _make_xl_model(self.tokenizers)
        positive = "a " * 225
        negative = "b " * 225
        medium_conditioning = model.combine_text_encoder_output(
            *model.encode_text(text=positive, train_device=torch.device("cpu"), max_caption_tokens=150)
        )
        positive_conditioning = model.combine_text_encoder_output(
            *model.encode_text(text=positive, train_device=torch.device("cpu"), max_caption_tokens=225)
        )
        negative_conditioning = model.combine_text_encoder_output(
            *model.encode_text(text=negative, train_device=torch.device("cpu"), max_caption_tokens=225)
        )
        self.assertEqual(medium_conditioning[0].shape, (1, 154, 40))
        self.assertEqual(medium_conditioning[1].shape, (1, 24))
        self.assertEqual(positive_conditioning[0].shape, (1, 231, 40))
        self.assertEqual(positive_conditioning[1].shape, (1, 24))
        self.assertEqual(negative_conditioning[0].shape, (1, 231, 40))
        self.assertEqual(negative_conditioning[1].shape, (1, 24))
        self.assertEqual(torch.cat([negative_conditioning[0], positive_conditioning[0]]).shape, (2, 231, 40))
        self.assertEqual(torch.cat([negative_conditioning[1], positive_conditioning[1]]).shape, (2, 24))

    def test_chunk_attention_mask_preserves_short_and_aligned_conditioning(self):
        tokenizer_1, tokenizer_2 = self.tokenizers
        captions = (
            ("", 1),
            ("a " * 40, 1),
            ("a " * 75, 1),
            ("a " * 76, 2),
            ("a " * 150, 2),
            ("a " * 151, 3),
            ("a " * 225, 3),
        )
        for caption, expected_chunks in captions:
            tokens_1, _ = tokenize_clip_chunks(tokenizer_1, caption, 225)
            tokens_2, _ = tokenize_clip_chunks(tokenizer_2, caption, 225)
            mask = sdxl_chunk_attention_mask(
                tokenizer_1, tokens_1, tokenizer_2, tokens_2, 225
            )
            self.assertEqual(mask.shape, (1, 231))
            self.assertTrue(torch.all(mask[0, :expected_chunks * 77] == 1))
            self.assertTrue(torch.all(mask[0, expected_chunks * 77:] == 0))

        short = "a " * 40
        short_1, _ = tokenize_clip_chunks(tokenizer_1, short, 225)
        short_2, _ = tokenize_clip_chunks(tokenizer_2, short, 225)
        legacy_1, _ = tokenize_clip_chunks(tokenizer_1, short, 75)
        legacy_2, _ = tokenize_clip_chunks(tokenizer_2, short, 75)
        self.assertTrue(torch.equal(short_1[:77], legacy_1))
        self.assertTrue(torch.equal(short_2[:77], legacy_2))

        aligned_l, _ = tokenize_clip_chunks(
            tokenizer_1, "a " * 151, 225
        )
        aligned_g, _ = tokenize_clip_chunks(
            tokenizer_2, "a " * 151, 225
        )
        aligned_g[2 * 77 + 1:] = tokenizer_2.pad_token_id
        aligned_g[2 * 77] = tokenizer_2.bos_token_id
        aligned_g[2 * 77 + 1] = tokenizer_2.eos_token_id
        aligned_mask = sdxl_chunk_attention_mask(
            tokenizer_1, aligned_l, tokenizer_2, aligned_g, 225
        )
        self.assertTrue(torch.all(aligned_mask[0, :2 * 77] == 1))
        self.assertTrue(torch.all(aligned_mask[0, 2 * 77:] == 0))

        short = "a " * 40
        long = "a " * 151
        batch_l = torch.stack([tokenize_clip_chunks(tokenizer_1, prompt, 225)[0] for prompt in (short, long)])
        batch_g = torch.stack([tokenize_clip_chunks(tokenizer_2, prompt, 225)[0] for prompt in (short, long)])
        batch_mask = sdxl_chunk_attention_mask(tokenizer_1, batch_l, tokenizer_2, batch_g, 225)
        self.assertEqual(batch_mask.shape, (2, 231))
        self.assertTrue(torch.all(batch_mask[0, :77] == 1))
        self.assertTrue(torch.all(batch_mask[0, 77:] == 0))
        self.assertTrue(torch.all(batch_mask[1, :3 * 77] == 1))
        self.assertTrue(torch.all(batch_mask[1, 3 * 77:] == 0))

    def test_pad_token_id_can_appear_as_caption_content(self):
        clip_l, clip_g = self.tokenizers
        caption = "a " * CLIP_CONTENT_TOKENS + "!"
        tokens_l, _ = tokenize_clip_chunks(clip_l, caption, 150)
        tokens_g, _ = tokenize_clip_chunks(clip_g, caption, 150)

        self.assertEqual(tokens_g[CLIP_CHUNK_TOKENS + 1].item(), clip_g.pad_token_id)
        mask = sdxl_chunk_attention_mask(clip_l, tokens_l, clip_g, tokens_g, 150)
        self.assertTrue(torch.all(mask == 1))

    def test_chunk_aware_unet_groups_mixed_lengths_and_restores_order(self):
        class RecordingUNet:
            def __init__(self):
                self.sequence_lengths = []

            def __call__(self, *, sample, timestep, encoder_hidden_states, added_cond_kwargs):
                self.sequence_lengths.append(encoder_hidden_states.shape[1])
                pooled = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
                time_condition = added_cond_kwargs["text_embeds"].mean(dim=1).view(-1, 1, 1, 1)
                return type("Output", (), {"sample": sample + pooled + time_condition + timestep.reshape(-1, 1, 1, 1)})()

        batch_size = 3
        hidden = torch.randn(batch_size, 231, 4, requires_grad=True)
        sample = torch.zeros(batch_size, 2, 2, 2)
        timestep = torch.tensor([5, 7, 11])
        pooled = torch.randn(batch_size, 6)
        lengths = (77, 231, 154)
        mask = torch.arange(231)[None, :] < torch.tensor(lengths)[:, None]
        unet = RecordingUNet()
        result = sdxl_chunk_aware_unet_prediction(
            unet,
            sample,
            timestep,
            hidden,
            {"text_embeds": pooled},
            mask,
        )
        self.assertEqual(sorted(unet.sequence_lengths), [77, 154, 231])
        expected = torch.cat([
            RecordingUNet()(
                sample=sample[i:i + 1], timestep=timestep[i:i + 1],
                encoder_hidden_states=hidden[i:i + 1, :length],
                added_cond_kwargs={"text_embeds": pooled[i:i + 1]},
            ).sample
            for i, length in enumerate(lengths)
        ])
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(torch.linalg.vector_norm(hidden.grad).item(), 0)

        short_hidden = torch.randn(2, 231, 4)
        short_sample = torch.randn(2, 2, 2, 2)
        short_pooled = torch.randn(2, 6)
        short_mask = mask[:1].expand(2, -1)
        extended_unet = RecordingUNet()
        extended_output = sdxl_chunk_aware_unet_prediction(
            extended_unet,
            short_sample,
            torch.tensor(5),
            short_hidden,
            {"text_embeds": short_pooled},
            short_mask,
        )
        legacy_unet = RecordingUNet()
        legacy_output = sdxl_chunk_aware_unet_prediction(
            legacy_unet,
            short_sample,
            torch.tensor(5),
            short_hidden[:, :77].contiguous(),
            {"text_embeds": short_pooled},
            None,
        )
        self.assertEqual(extended_unet.sequence_lengths, [77])
        torch.testing.assert_close(extended_output, legacy_output, rtol=0, atol=0)

    def test_short_caption_hidden_states_and_empty_prompt_mask(self):
        model = _make_xl_model(self.tokenizers)
        short = "a " * 40
        legacy = model.encode_text(text=short, train_device=torch.device("cpu"))
        extended = model.encode_text(
            text=short,
            train_device=torch.device("cpu"),
            max_caption_tokens=225,
            return_chunk_attention_mask=True,
        )
        for legacy_output, extended_output in zip(legacy, extended[:3], strict=True):
            torch.testing.assert_close(legacy_output, extended_output[..., :legacy_output.shape[-2], :]
                                       if legacy_output.ndim == 3 else extended_output)
        self.assertEqual(extended[3].shape, (1, 231))
        self.assertTrue(torch.all(extended[3][:, :77] == 1))
        self.assertTrue(torch.all(extended[3][:, 77:] == 0))

        empty = model.encode_text(
            text="",
            train_device=torch.device("cpu"),
            max_caption_tokens=225,
            return_chunk_attention_mask=True,
        )
        self.assertTrue(torch.all(empty[3][:, :77] == 1))
        self.assertTrue(torch.all(empty[3][:, 77:] == 0))

    def test_75_token_conditioning_matches_legacy_encoder_calls(self):
        torch.manual_seed(123)
        encoder_1, encoder_2 = _small_clip_encoders(self.tokenizers)
        prompt = "a " * 74
        tokens_1, _ = tokenize_clip_chunks(self.tokenizers[0], prompt, 75)
        tokens_2, _ = tokenize_clip_chunks(self.tokenizers[1], prompt, 75)

        legacy_1 = encode_clip(
            encoder_1, tokens_1.unsqueeze(0), default_layer=-2, add_pooled_output=False,
            use_attention_mask=False, add_layer_norm=False,
        )[0]
        chunked_1, _ = encode_clip_chunks(encoder_1, tokens_1, hidden_state_output_index=-2)
        legacy_2 = encode_clip(
            encoder_2, tokens_2.unsqueeze(0), default_layer=-2, add_pooled_output=True,
            use_attention_mask=False, add_layer_norm=False,
        )
        chunked_2, chunked_pool = encode_clip_chunks(
            encoder_2, tokens_2, hidden_state_output_index=-2, add_pooled_output=True,
        )
        torch.testing.assert_close(chunked_1, legacy_1, rtol=0, atol=0)
        torch.testing.assert_close(chunked_2, legacy_2[0], rtol=0, atol=0)
        torch.testing.assert_close(chunked_pool, legacy_2[1], rtol=0, atol=0)

    def test_trainable_text_encoder_retains_gradient_flow(self):
        model = _make_xl_model(self.tokenizers)
        tokens_1, _ = tokenize_clip_chunks(self.tokenizers[0], "a " * 151, 225)
        tokens_2, _ = tokenize_clip_chunks(self.tokenizers[1], "a " * 151, 225)
        output = model.encode_text(
            train_device=torch.device("cpu"),
            tokens_1=tokens_1.unsqueeze(0),
            tokens_2=tokens_2.unsqueeze(0),
            max_caption_tokens=225,
            return_chunk_attention_mask=True,
        )
        combined, pooled = model.combine_text_encoder_output(*output[:3])
        self.assertTrue(torch.all(output[3] == 1))
        (combined.square().mean() + pooled.square().mean()).backward()
        gradient = model.text_encoder_1.text_model.encoder.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(torch.linalg.vector_norm(gradient).item(), 0)

        with self.assertRaisesRegex(ValueError, "expected 77"):
            model.encode_text(
                train_device=torch.device("cpu"),
                tokens_1=tokens_1.unsqueeze(0),
                tokens_2=tokens_2.unsqueeze(0),
                max_caption_tokens=75,
            )


if __name__ == "__main__":
    unittest.main()
