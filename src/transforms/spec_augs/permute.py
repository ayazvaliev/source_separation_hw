import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, permutation: tuple[int], **kwargs):
        super().__init__()
        self.permutation = permutation

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return spectrogram.permute(*self.permutation).contiguous()
