import huggingface_hub
import open_clip
import torch
from torch import nn
from torchvision.transforms import transforms


class HPSv2ScoreModel(nn.Module):
    def __init__(
            self,
            dtype: torch.dtype,
    ):
        super(HPSv2ScoreModel, self).__init__()
        self.dtype = dtype

        self.model = self.__load_open_clip_model()
        self.tokenizer = self.__load_tokenizer()

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])
        self.resize = transforms.Resize(224)
        self.crop = transforms.CenterCrop(224)


    def __load_open_clip_model(self):
        model_name = "ViT-H-14"

        precision = "fp32"
        match self.dtype:
            case torch.float16:
                precision = "fp16"
            case torch.bfloat16:
                precision = "bf16"

        open_clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=precision,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
        )

        model_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", "HPS_v2_compressed.pt"
        )
        checkpoint = torch.load(model_path)
        open_clip_model.load_state_dict(checkpoint['state_dict'])

        return open_clip_model

    def __load_tokenizer(self):
        model_name = "ViT-H-14"
        return open_clip.get_tokenizer(model_name)

    def forward(self, x, prompt, device):
        x = (x / 2.0 + 0.5).clamp(0.0, 1.0)
        x = self.crop(self.resize(x))
        x = self.normalize(x)

        caption = self.tokenizer(prompt)
        caption = caption.to(device)
        outputs = self.model(x, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        return 1.0 - scores
