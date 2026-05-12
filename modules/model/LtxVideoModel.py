import torch
import torch.nn as nn
from diffusers import LTXPipeline, AutoencoderKLLTX, LTXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer
from modules.model.BaseModel import BaseModel
from modules.util import list_utils
from modules.util.TrainProgress import TrainProgress


class LtxVideoModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model: LTXTransformer3DModel = None
        self.vae: AutoencoderKLLTX = None
        self.text_encoder: T5EncoderModel = None
        self.tokenizer: T5Tokenizer = None
        self.pipeline: LTXPipeline = None

    def load_model(self, model_path, vae_path=None, text_encoder_path=None, weight_dtype=torch.float32):
        self.pipeline = LTXPipeline.from_pretrained(model_path, torch_dtype=weight_dtype)
        self.model = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer

        if vae_path:
            self.vae = AutoencoderKLLTX.from_pretrained(vae_path, torch_dtype=weight_dtype)
        if text_encoder_path:
            self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=weight_dtype)

        self.model.requires_grad_(True)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def encode(self, video_frames, frames_indices=None):
        batch_size, num_frames, channels, height, width = video_frames.shape
        video_frames = video_frames.to(self.vae.device, dtype=self.vae.dtype)
        latents = []
        for t in range(num_frames):
            frame = video_frames[:, t, :, :, :]
            latent = self.vae.encode(frame).latent_dist.sample()
            latents.append(latent)
        latents = torch.stack(latents, dim=1)
        return latents * self.vae.config.scaling_factor

    def get_noise_prediction(self, latents, timestep, conditioning, mask=None):
        latents = latents.to(self.model.device, dtype=self.model.dtype)
        timestep = timestep.to(self.model.device)
        conditioning = conditioning.to(self.model.device, dtype=self.model.dtype)
        noise_pred = self.model(
            latent_sample=latents,
            timestep=timestep,
            encoder_hidden_states=conditioning,
            return_dict=False
        )[0]
        return noise_pred

    def get_target(self, latents, noise):
        return noise

    def get_conditioning(self, prompt, negative_prompt=None, prompt_2=None, negative_prompt_2=None):
        text_inputs = self.tokenizer(prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.text_encoder.device))[0]
        if negative_prompt:
            neg_inputs = self.tokenizer(negative_prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            neg_embeddings = self.text_encoder(neg_inputs.input_ids.to(self.text_encoder.device))[0]
            return torch.cat([neg_embeddings, text_embeddings])
        return text_embeddings

    def decode_latent(self, latents, frames_indices=None):
        latents = latents / self.vae.config.scaling_factor
        batch_size, num_frames, channels, height, width = latents.shape
        decoded_frames = []
        for t in range(num_frames):
            frame_latent = latents[:, t, :, :, :]
            decoded = self.vae.decode(frame_latent).sample
            decoded_frames.append(decoded)
        decoded_frames = torch.stack(decoded_frames, dim=1)
        return decoded_frames.clamp(-1, 1)

    def get_latent_shape(self, batch_size, num_frames, height, width):
        latent_height = height // self.vae.config.downsampling_factor
        latent_width = width // self.vae.config.downsampling_factor
        return (batch_size, num_frames, self.vae.config.latent_channels, latent_height, latent_width)

    def apply_lora(self, lora_path, alpha=1.0, target_modules=None):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=int(alpha * 16),
            lora_alpha=alpha,
            target_modules=target_modules or ["to_q", "to_k", "to_v", "to_out"],
            lora_dropout=0.0,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        lora_state = torch.load(lora_path, map_location='cpu')
        self.model.load_state_dict(lora_state, strict=False)

    def save_lora(self, output_path):
        lora_state = {k: v for k, v in self.model.state_dict().items() if 'lora' in k}
        torch.save(lora_state, output_path)

    def to(self, device, dtype=None):
        self.model.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.text_encoder.to(device, dtype=dtype)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
