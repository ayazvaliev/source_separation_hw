import torch
import torch.nn as nn
from torch_audiomentations import ApplyImpulseResponse


class ImpulseResponse(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ir_transform = ApplyImpulseResponse(**kwargs)

    def __call__(self, audio: torch.Tensor, **kwargs):
        audio = self.ir_transform(audio)
        return audio
