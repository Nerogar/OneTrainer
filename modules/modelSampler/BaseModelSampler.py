import os
import io
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.VideoFormat import VideoFormat

import torch
from torchvision.io import write_video

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
        if self.file_type == FileType.IMAGE:
            b = io.BytesIO()
            self.data.save(b, format='JPEG')
            return (ModelSamplerOutput, (self.file_type, b.getvalue()))
        elif self.file_type == FileType.VIDEO:
            #do not transfer videos; they are not shown anyway
            #the video sample file is transferred via workspace sync
            return (ModelSamplerOutput, (self.file_type, None))
        elif sampler_output.file_type == FileType.AUDIO:
            pass #TODO


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

    @staticmethod
    def quantize_resolution(resolution: int, quantization: int) -> int:
        return round(resolution / quantization) * quantization

    @staticmethod
    def save_sampler_output(
            sampler_output: ModelSamplerOutput,
            destination: str,
            image_format: ImageFormat,
            video_format: VideoFormat,
            audio_format: AudioFormat,
    ):
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        if sampler_output.file_type == FileType.IMAGE:
            image = sampler_output.data
            image.save(destination + image_format.extension(), format=image_format.pil_format())
        elif sampler_output.file_type == FileType.VIDEO:
            write_video(destination + video_format.extension(), options={"crf": "17"}, video_array=sampler_output.data, fps=24)
        elif sampler_output.file_type == FileType.AUDIO:
            pass # TODO
