import torch
import torch.nn as nn
from torch_audiomentations import AddColoredNoise


class ColouredNoise(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.coloured_noise_transform = AddColoredNoise(**kwargs)

    def __call__(self, audio: torch.Tensor, **kwargs):
        audio = self.coloured_noise_transform(audio)
        return audio
