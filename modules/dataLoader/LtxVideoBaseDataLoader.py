import torch
from torch.utils.data import DataLoader
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.LtxVideoModel import LtxVideoModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType

class LtxVideoBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def __init__(self, train_progress: TrainProgress, config: TrainConfig, model: LtxVideoModel, device: torch.device):
        super().__init__(train_progress, config, model, device)
        self.frame_rate = 25
        self.latent_scaling_factor = model.latent_scaling_factor if hasattr(model, 'latent_scaling_factor') else 0.18215

    def load_batch(self, batch: dict) -> dict:
        video_path = batch['video_path']
        frames = self._load_video_frames(video_path, batch.get('start_frame', 0), batch.get('num_frames', 16))
        latents = batch.get('latents')
        if latents is None:
            latents = self.encode_latents(frames)
        else:
            latents = latents.to(self.device)
        conditioning = self.encode_prompts(batch.get('caption', ''))
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(self.device)
        return {
            'latents': latents,
            'conditioning': conditioning,
            'mask': mask,
            'frames': frames.to(self.device) if batch.get('return_frames', False) else None
        }

    def encode_prompts(self, caption: str) -> dict:
        tokenizer = self.model.tokenizer
        text_encoder = self.model.text_encoder
        tokens = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=77).to(self.device)
        with torch.no_grad():
            encoder_hidden_states = text_encoder(tokens.input_ids)[0]
        return {'encoder_hidden_states': encoder_hidden_states, 'attention_mask': tokens.attention_mask}

    def encode_latents(self, frames: torch.Tensor) -> torch.Tensor:
        vae = self.model.vae
        frames = frames.to(self.device)
        with torch.no_grad():
            latents = vae.encode(frames).latent_dist.sample() * self.latent_scaling_factor
        return latents

    def _load_video_frames(self, video_path: str, start_frame: int, num_frames: int) -> torch.Tensor:
        import decord
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        end_frame = min(start_frame + num_frames, total_frames)
        indices = list(range(start_frame, end_frame))
        if len(indices) < num_frames:
            indices = indices + [indices[-1]] * (num_frames - len(indices))
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).float() / 127.5 - 1.0
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def __getitem__(self, index):
        sample = self.dataset[index]
        batch = self.load_batch(sample)
        return batch

    def collate_fn(self, batch):
        latents = torch.stack([item['latents'] for item in batch])
        conditioning = {'encoder_hidden_states': torch.stack([item['conditioning']['encoder_hidden_states'] for item in batch]),
                        'attention_mask': torch.stack([item['conditioning']['attention_mask'] for item in batch])}
        mask = torch.stack([item['mask'] for item in batch]) if batch[0]['mask'] is not None else None
        frames = torch.stack([item['frames'] for item in batch]) if batch[0]['frames'] is not None else None
        return {'latents': latents, 'conditioning': conditioning, 'mask': mask, 'frames': frames}
