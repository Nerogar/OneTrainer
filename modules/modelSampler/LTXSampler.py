import copy
from collections.abc import Callable
from contextlib import nullcontext

from modules.model.LTXModel import LTXModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.SamplingMethod import SamplingMethod
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.staged_pipeline import run_staged_pipeline
from modules.util.tqdm_util import tqdm

import torch

import numpy as np

# The distilled expert's own fixed sigma schedule (Lightricks' DISTILLED_SIGMA_VALUES). Literal values, never
# run through calculate_timestep_shift: the reference passes them straight to the stage, bypassing the
# token-count shift.
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)

# The tail of that schedule (Lightricks' STAGE_2_DISTILLED_SIGMA_VALUES), what the expert runs when it takes
# over a trajectory already denoised down to its first value - which doubles as the hand-off point.
LOW_NOISE_SIGMAS = DISTILLED_SIGMAS[5:]

# The sigma the full-model schedule ends on in both generations. 2.3's scheduler carries it as shift_terminal,
# 2.5's ships null but only because that scheduler is configured for the distilled DiT, which walks its own
# fixed sigma list instead.
SHIFT_TERMINAL = 0.1


@factory.register(BaseModelSampler, ModelType.LTX_2)
class LTXSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: LTXModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def __text_encode(
            self,
            sample_config: SampleConfig,
    ) -> dict:
        do_cfg = sample_config.cfg_scale > 1.0

        self.model.materialize_only("text_encoder")
        text_encoder_outputs, tokens_mask = self.model.encode_text_encoder(
            text=[sample_config.negative_prompt, sample_config.prompt] if do_cfg else sample_config.prompt,
        )

        # park the TE outputs in CPU RAM: this stage runs over the whole batch before the next one starts, and
        # the stacked outputs are large enough to fill VRAM at a higher batch size
        text_encoder_outputs = [output.to("cpu") for output in text_encoder_outputs]
        return {
            "text_encoder_outputs": text_encoder_outputs,
            "tokens_mask": tokens_mask.to("cpu"),
        }

    @torch.no_grad()
    def __connect(
            self,
            text_encoder_outputs: list[torch.Tensor],
            tokens_mask: torch.Tensor,
    ) -> dict:
        self.model.materialize_only("connectors")
        connector_prompt_embeds, connector_audio_prompt_embeds = self.model.encode_connectors(
            text_encoder_outputs, tokens_mask, self.train_device,
        )

        return {
            "connector_prompt_embeds": connector_prompt_embeds,
            "connector_audio_prompt_embeds": connector_audio_prompt_embeds,
        }

    @torch.no_grad()
    def __run_denoise_loop(
            self,
            transformer,
            autocast_context,
            dtype: torch.dtype,
            timesteps: torch.Tensor,
            noise_scheduler,
            audio_noise_scheduler,
            latent_video: torch.Tensor,
            latent_audio: torch.Tensor,
            geometry: dict,
            conditioning: dict,
            do_cfg: bool,
            cfg_scale: float,
            on_update_progress: Callable[[int, int], None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # run once per expert - both share an architecture, so only the module, its autocast context and its
        # compute dtype differ
        # with CFG the whole input side is a [negative, positive] batch: the embeds already arrive stacked that
        # way, the latents and the coords are duplicated to match
        video_coords = torch.cat([geometry["video_coords"]] * 2) if do_cfg else geometry["video_coords"]
        audio_coords = torch.cat([geometry["audio_coords"]] * 2) if do_cfg else geometry["audio_coords"]

        for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):
            latent_video_input = torch.cat([latent_video] * 2) if do_cfg else latent_video
            latent_audio_input = torch.cat([latent_audio] * 2) if do_cfg else latent_audio
            expanded_timestep = timestep.expand(latent_video_input.shape[0])

            with autocast_context:
                noise_pred_video, noise_pred_audio = transformer(
                    hidden_states=latent_video_input.to(dtype=dtype),
                    audio_hidden_states=latent_audio_input.to(dtype=dtype),
                    encoder_hidden_states=conditioning["connector_prompt_embeds"].to(dtype=dtype),
                    audio_encoder_hidden_states=conditioning["connector_audio_prompt_embeds"].to(dtype=dtype),
                    timestep=expanded_timestep,
                    sigma=expanded_timestep,
                    encoder_attention_mask=None,
                    audio_encoder_attention_mask=None,
                    num_frames=geometry["num_latent_frames"],
                    height=geometry["latent_height"],
                    width=geometry["latent_width"],
                    fps=geometry["frame_rate"],
                    audio_num_frames=geometry["audio_num_frames"],
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    return_dict=False,
                )

            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            if do_cfg:
                noise_pred_video_uncond, noise_pred_video_cond = noise_pred_video.chunk(2)
                noise_pred_video = noise_pred_video_uncond \
                    + cfg_scale * (noise_pred_video_cond - noise_pred_video_uncond)

                noise_pred_audio_uncond, noise_pred_audio_cond = noise_pred_audio.chunk(2)
                noise_pred_audio = noise_pred_audio_uncond \
                    + cfg_scale * (noise_pred_audio_cond - noise_pred_audio_uncond)

            latent_video = noise_scheduler.step(noise_pred_video, timestep, latent_video, return_dict=False)[0]
            # audio branch is never decoded (video-only scope) - only stepped so the transformer keeps seeing a
            # properly noised audio trajectory, matching what the model saw during audio-visual training
            latent_audio = audio_noise_scheduler.step(
                noise_pred_audio, timestep, latent_audio, return_dict=False,
            )[0]

            on_update_progress(i + 1, len(timesteps))

        return latent_video, latent_audio

    @torch.no_grad()
    def __denoise(
            self,
            sample_config: SampleConfig,
            connector_prompt_embeds: torch.Tensor,
            connector_audio_prompt_embeds: torch.Tensor,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        cfg_scale = sample_config.cfg_scale
        diffusion_steps = sample_config.diffusion_steps
        do_cfg = cfg_scale > 1.0

        num_frames = self.quantize_resolution(sample_config.frames - 1, 8) + 1

        generator = torch.Generator(device=self.train_device)
        if sample_config.random_seed:
            generator.seed()
        else:
            generator.manual_seed(sample_config.seed)

        noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
        # LTX-2 denoises the audio branch alongside the video branch even in video-only use, so it needs its
        # own scheduler instance (mirrors LTX2Pipeline.__call__'s `audio_scheduler`)
        audio_noise_scheduler = copy.deepcopy(self.model.noise_scheduler)
        transformer = self.pipeline.transformer
        audio_vae = self.pipeline.audio_vae

        frame_rate = self.model.NATIVE_FPS
        vae_spatial_scale_factor = self.pipeline.vae_spatial_compression_ratio

        num_latent_frames = (num_frames - 1) // self.pipeline.vae_temporal_compression_ratio + 1
        latent_height = self.quantize_resolution(sample_config.height, 32) // vae_spatial_scale_factor
        latent_width = self.quantize_resolution(sample_config.width, 32) // vae_spatial_scale_factor
        latent_video = torch.randn(
            size=(1, transformer.config.in_channels, num_latent_frames, latent_height, latent_width),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32,
        )
        latent_video = self.model.pack_latents(
            latent_video,
            self.pipeline.transformer_spatial_patch_size, self.pipeline.transformer_temporal_patch_size,
        )

        # prepare audio latents: fed real noise only because the transformer takes audio inputs positionally,
        # denoised in lockstep to mirror the reference pipeline but never decoded
        audio_num_frames = round(
            num_frames / frame_rate
            * self.pipeline.audio_sampling_rate
            / self.pipeline.audio_hop_length
            / float(self.pipeline.audio_vae_temporal_compression_ratio)
        )
        latent_audio = torch.randn(
            size=(1, audio_vae.config.latent_channels, audio_num_frames,
                  audio_vae.config.mel_bins // self.pipeline.audio_vae_mel_compression_ratio),
            generator=generator,
            device=self.train_device,
            dtype=torch.float32,
        )
        latent_audio = self.model.pack_audio_latents(latent_audio)

        if sample_config.override_shift:
            shift = sample_config.override_shift
        else:
            shift = self.model.calculate_timestep_shift(num_latent_frames, latent_height, latent_width)

        # the reference's schedule: a 1/steps grid, bent towards the noisy end by the token-count shift, then
        # stretched so its last sigma lands on SHIFT_TERMINAL
        sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps, dtype=np.float32)
        sigmas = shift / (shift + (1.0 / sigmas - 1.0))
        sigmas = 1.0 - (1.0 - sigmas) * (1.0 - SHIFT_TERMINAL) / (1.0 - sigmas[-1])

        # one schedule spans both stages, split at expert_start: this stage walks the steps before it, the
        # expert stage continues on the same schedulers from there
        if self.model.low_noise_transformer is not None:
            if sample_config.sampling_method == SamplingMethod.HANDOFF_LOW_NOISE:
                # the expert takes over at its own first sigma, so this stage keeps the steps above it
                above_handoff = sigmas[sigmas > LOW_NOISE_SIGMAS[0]]
                expert_start = len(above_handoff)
                sigmas = np.append(above_handoff, LOW_NOISE_SIGMAS[:-1])
            elif sample_config.sampling_method == SamplingMethod.DISTILLED:
                # the expert samples from noise on its own schedule, so it walks the whole trajectory
                expert_start = 0
                sigmas = np.asarray(DISTILLED_SIGMAS[:-1], dtype=np.float32)
            elif sample_config.sampling_method == SamplingMethod.STANDARD:
                expert_start = len(sigmas)
            else:
                raise NotImplementedError(f"unsupported sampling method {sample_config.sampling_method}")
        else:
            # nothing to hand over to, so a method that wants the expert runs the trained transformer alone
            expert_start = len(sigmas)

        # the schedule is installed verbatim, so the scheduler only does the Euler integration
        for scheduler in (noise_scheduler, audio_noise_scheduler):
            scheduler.register_to_config(use_dynamic_shifting=False, shift=1.0, shift_terminal=None)
            scheduler.set_timesteps(sigmas=sigmas.astype(np.float32), device=self.train_device)

        timesteps = noise_scheduler.timesteps[:expert_start]

        # the steps walked here are not the configured count: a hand-off drops everything below its sigma
        # to the expert, and a larger shift pushes more of the schedule above it, so more steps stay here.
        # Printed in timestep units (sigma * 1000) so it lines up with the schedule.
        tqdm.write(f"[sigmas] denoise: shift {shift:.3f}, {diffusion_steps} configured -> "
                   f"{len(timesteps)} steps, timesteps {[round(float(t), 1) for t in timesteps]}")

        video_coords = transformer.rope.prepare_video_coords(
            latent_video.shape[0], num_latent_frames, latent_height, latent_width, latent_video.device,
            fps=frame_rate,
        )
        audio_coords = transformer.audio_rope.prepare_audio_coords(
            latent_audio.shape[0], audio_num_frames, latent_audio.device,
        )

        geometry = {
            "num_latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "audio_num_frames": audio_num_frames,
            "video_coords": video_coords,
            "audio_coords": audio_coords,
            "frame_rate": frame_rate,
        }
        conditioning = {
            "connector_prompt_embeds": connector_prompt_embeds,
            "connector_audio_prompt_embeds": connector_audio_prompt_embeds,
        }
        if len(timesteps) > 0:
            # skipped in DISTILLED, where this stage takes no step - the 19B trained transformer then never
            # reaches the train device at all
            self.model.materialize_only("transformer")
            latent_video, latent_audio = self.__run_denoise_loop(
                transformer, self.model.transformer_autocast_context,
                self.model.transformer_train_dtype.torch_dtype(), timesteps,
                noise_scheduler, audio_noise_scheduler, latent_video, latent_audio,
                geometry, conditioning, do_cfg, cfg_scale, on_update_progress,
            )

        if do_cfg:
            # __text_encode stacks the embeds as [negative, positive]; the low noise expert stage runs
            # unguided, so it is handed the conditional half alone
            conditioning = {name: tensor.chunk(2)[1] for name, tensor in conditioning.items()}

        return {
            "latent_video": latent_video,
            "latent_audio": latent_audio,
            "num_latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "geometry": geometry,
            "conditioning": conditioning,
            "noise_scheduler": noise_scheduler,
            "audio_noise_scheduler": audio_noise_scheduler,
            "expert_timesteps": noise_scheduler.timesteps[expert_start:],
        }

    @torch.no_grad()
    def __denoise_low_noise(
            self,
            latent_video: torch.Tensor,
            latent_audio: torch.Tensor,
            noise_scheduler,
            audio_noise_scheduler,
            expert_timesteps: torch.Tensor,
            geometry: dict,
            conditioning: dict,
            on_update_progress: Callable[[int, int], None],
    ) -> dict:
        # the split left nothing over, so the trained transformer already walked the whole schedule
        if len(expert_timesteps) == 0:
            return {"latent_video": latent_video}

        # printed next to the other stage's line to make the whole trajectory visible at once
        tqdm.write(f"[sigmas] low noise expert: {len(expert_timesteps)} steps, "
                   f"timesteps {[round(float(t), 1) for t in expert_timesteps]}")

        self.model.materialize_only("low_noise_transformer")
        dtype = self.model.low_noise_transformer_train_dtype.torch_dtype()

        if self.model.transformer_lora is None:
            lora_context = nullcontext()
        else:
            # the same LoRA weights, rebound onto the expert for this stage. A LoRA travels with its stem, so
            # materializing the expert evicted it along with the transformer - bring it back on its own, or
            # the expert's forward mixes cuda and cpu tensors. The next materialize of either part takes it
            # along again.
            self.model.transformer_lora.to(device=self.train_device)
            lora_context = self.model.transformer_lora.retargeted(self.model.low_noise_transformer)

        with lora_context:
            latent_video, _ = self.__run_denoise_loop(
                self.model.low_noise_transformer, self.model.low_noise_transformer_autocast_context, dtype,
                expert_timesteps, noise_scheduler, audio_noise_scheduler,
                latent_video, latent_audio, geometry, conditioning, False, 1.0, on_update_progress,
            )

        return {"latent_video": latent_video}

    @torch.no_grad()
    def __decode(
            self,
            latent_video: torch.Tensor,
            num_latent_frames: int,
            latent_height: int,
            latent_width: int,
    ) -> ModelSamplerOutput:
        # evict the transformer and materialize only the vae
        self.model.materialize_only("vae")
        vae = self.pipeline.vae
        patch_size = self.pipeline.transformer_spatial_patch_size
        patch_size_t = self.pipeline.transformer_temporal_patch_size

        latent_video = self.model.unpack_latents(
            latent_video, num_latent_frames, latent_height, latent_width, patch_size, patch_size_t,
        )
        latent_video = self.model.unscale_latents(latent_video)
        latent_video = latent_video.to(dtype=vae.dtype)

        # a timestep-conditioned decoder would need a per-item timestep for its temb; both LTX-2 VAEs decode
        # unconditionally, so none is built
        assert not vae.config.timestep_conditioning

        video = vae.decode(latent_video, None, return_dict=False)[0]
        # postprocess_video permutes frames ahead of channels, landing on [B, F, C, H, W]
        video = self.pipeline.video_processor.postprocess_video(video, output_type='pt')

        return self.build_video_sampler_output(video)

    def sample_all(
            self,
            sample_configs: list[SampleConfig],
            destinations: list[str],
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> list[ModelSamplerOutput]:
        batch_progress = self.batch_progress_callback(sample_configs, on_update_progress)

        with self.model.autocast_context:
            sampler_outputs = run_staged_pipeline(
                [("encoding text", self.__text_encode), ("connecting", self.__connect),
                 ("denoising", self.__denoise), ("denoising low noise", self.__denoise_low_noise),
                 ("decoding", self.__decode)],
                {"sample_config": sample_configs},
                {"on_update_progress": batch_progress},
            )

        for sampler_output, destination in zip(sampler_outputs, destinations, strict=True):
            self.save_sampler_output(
                sampler_output, destination,
                image_format, video_format, audio_format,
                fps=self.model.NATIVE_FPS,
            )

        return sampler_outputs

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        # single-sample entry point: a staged batch of one
        sampler_output = self.sample_all(
            [sample_config], [destination],
            image_format, video_format, audio_format,
            on_update_progress=on_update_progress,
        )[0]

        on_sample(sampler_output)
