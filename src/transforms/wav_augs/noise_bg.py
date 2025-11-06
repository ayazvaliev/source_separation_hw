import torch
import torch.nn as nn
from torch_audiomentations import AddBackgroundNoise


class BackgroundNoise(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bg_transform = AddBackgroundNoise(**kwargs)

    def __call__(self, audio: torch.Tensor):
        audio = self.bg_transform(audio)
        return audio
