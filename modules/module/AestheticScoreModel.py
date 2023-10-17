import os

import pooch
import torch
from torch import nn
from torchvision.transforms import transforms
from transformers import CLIPModel


class MLPModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MLPModel, self).__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScoreModel(nn.Module):
    def __init__(
            self,
    ):
        super(AestheticScoreModel, self).__init__()

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp_model = self.__load_mlp_model()

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])
        self.resize = transforms.Resize(224)
        self.crop = transforms.CenterCrop(224)

        self.score_target = 10.0


    def __load_mlp_model(self):
        filename = "sac+logos+ava1-l14-linearMSE.pth"
        path = os.path.join("external", "models", "laionAesthetics")

        pooch.retrieve(
            "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
            "md5:b1047fd767a00134b8fd6529bf19521a",
            fname=filename,
            path=path,
            progressbar=True,
        )

        mlp_model = MLPModel()
        mlp_model.load_state_dict(torch.load(os.path.join(path, filename)))

        return mlp_model

    def forward(self, x):
        x = (x / 2.0 + 0.5).clamp(0.0, 1.0)
        x = self.crop(self.resize(x))
        x = self.normalize(x)
        embedding = self.clip.get_image_features(pixel_values=x)
        embedding = embedding / torch.linalg.vector_norm(embedding, dim=-1, keepdim=True)
        score = self.mlp_model(embedding).squeeze(1)
        return abs(score - self.score_target)
