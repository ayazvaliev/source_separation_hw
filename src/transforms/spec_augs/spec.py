import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, Spectrogram


class LogSpecTransform(nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_length: int, power: float, **kwargs):
        super().__init__()
        self.spec_transform = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
        )
        self.amplitude_to_db = AmplitudeToDB("power" if power == 2.0 else "magnitude", top_db=80)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.amplitude_to_db(self.spec_transform(audio.squeeze(0)))
