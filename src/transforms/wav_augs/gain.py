import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.gain_transform = torch_audiomentations.Gain(**kwargs)

    def __call__(self, audio: Tensor):
        return self.gain_transform(audio)
