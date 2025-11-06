import torch
import torch.nn as nn


class MeanNormalization(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return spectrogram - spectrogram.mean()
