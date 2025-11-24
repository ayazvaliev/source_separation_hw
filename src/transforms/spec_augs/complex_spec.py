import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram


class ComplexSpec(nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int, center: bool, **kwargs):
        super().__init__()
        self.spec_transform = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None, center=center
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        complex_spec = self.spec_transform(audio)
        return torch.stack([torch.real(complex_spec), torch.imag(complex_spec)], dim=1)
