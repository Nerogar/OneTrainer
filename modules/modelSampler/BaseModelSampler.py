import io
import os
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.VideoFormat import VideoFormat

import torch

import av
from PIL import Image


class ModelSamplerOutput:
    def __init__(
            self,
            file_type: FileType,
            data: Image.Image | torch.Tensor | bytes,

    ):
        self.file_type = file_type
        if isinstance(data, bytes):
            assert file_type == FileType.IMAGE
            self.data = Image.open(io.BytesIO(data))
        else:
            self.data = data

    #Reduce to a JPEG bytestream for cloud training:
    def __reduce__(self):
        match self.file_type:
            case FileType.IMAGE:
                b = io.BytesIO()
                self.data.save(b, format='JPEG')
                return ModelSamplerOutput, (self.file_type, b.getvalue())
            case FileType.VIDEO:
                #do not transfer videos; they are not shown anyway
                #the video sample file is transferred via workspace sync
                return ModelSamplerOutput, (self.file_type, None)
            case FileType.AUDIO:
                # TODO
                return ModelSamplerOutput, (self.file_type, None)
            case _:
                return ModelSamplerOutput, (self.file_type, None)


class BaseModelSampler(metaclass=ABCMeta):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device

    @abstractmethod
    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat,
            video_format: VideoFormat,
            audio_format: AudioFormat,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        pass

    def sample_all(
            self,
            sample_configs: list[SampleConfig],
            destinations: list[str],
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> list[ModelSamplerOutput]:
        # Item-major fallback for samplers not split into pipeline stages: run each sample
        # end to end, cycling this model's parts on-device once per sample instead of once
        # per batch. Progress restarts at each sample, matching how these samplers reported
        # when the trainer still sampled one at a time. Samplers that define stages override
        # this with run_staged_pipeline.
        sampler_outputs = []
        for sample_config, destination in zip(sample_configs, destinations, strict=True):
            self.sample(
                sample_config, destination,
                image_format, video_format, audio_format,
                on_sample=sampler_outputs.append,
                on_update_progress=on_update_progress,
            )
        return sampler_outputs

    @staticmethod
    def quantize_resolution(resolution: int, quantization: int) -> int:
        return round(resolution / quantization) * quantization

    @staticmethod
    def batch_progress_callback(
            sample_configs: list[SampleConfig],
            on_update_progress: Callable[[int, int], None],
    ) -> Callable[[int, int], None]:
        # denoise reports its own per-sample step count; the staged pipeline denoises
        # every sample before any decode, so translate those into one continuous bar
        # spanning the whole batch rather than restarting at each sample
        total_steps = sum(sample_config.diffusion_steps for sample_config in sample_configs)
        completed_steps = 0

        def batch_progress(step: int, sample_steps: int):
            nonlocal completed_steps
            on_update_progress(completed_steps + step, total_steps)
            if step == sample_steps:
                completed_steps += sample_steps

        return batch_progress

    @staticmethod
    def save_sampler_output(
            sampler_output: ModelSamplerOutput,
            destination: str,
            image_format: ImageFormat | None,
            video_format: VideoFormat | None,
            audio_format: AudioFormat | None,
            fps: int = 24,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if sampler_output.file_type == FileType.IMAGE:
            if image_format is None:
                raise ValueError("Image format required for sampling an image")
            image = sampler_output.data
            image.save(destination + image_format.extension(), format=image_format.pil_format())
        elif sampler_output.file_type == FileType.VIDEO:
            if video_format is None:
                raise ValueError("Video format required for sampling a video")

            if isinstance(sampler_output.data, torch.Tensor):
                video_tensor = sampler_output.data.detach().cpu()

                if len(video_tensor.shape) == 4:
                    shape = video_tensor.shape
                    # (T, H, W, C) if last dim is channels, otherwise assume (C, T, H, W)
                    frames = video_tensor.numpy() if shape[-1] == 3 else video_tensor.permute(1, 2, 3, 0).numpy()

                    frames = (
                        (frames * 255).astype('uint8')
                        if frames.max() <= 1.0
                        else frames.astype('uint8')
                    )

                    with av.open(destination + video_format.extension(), 'w') as container:
                        stream = container.add_stream('libx264', rate=fps)
                        stream.options = {'crf': '17'}
                        stream.width = frames.shape[2]
                        stream.height = frames.shape[1]
                        stream.pix_fmt = 'yuv420p'  # Required pixel format for H.264

                        for frame_data in frames:
                            frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                            for packet in stream.encode(frame):
                                container.mux(packet)

                        for packet in stream.encode():
                            container.mux(packet)
                else:
                    raise ValueError(f"Expected 4D video tensor (T, H, W, C) or (C, T, H, W), got shape {video_tensor.shape}")
        elif sampler_output.file_type == FileType.AUDIO:
            pass # TODO
