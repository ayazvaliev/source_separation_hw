import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class LogMelSpecTransform(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float,
        top_db: float,
        **kwargs,
    ):
        super().__init__()
        self.melspec_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        )
        assert top_db >= 0
        self.top_db = top_db
        self.amin = torch.tensor(1e-10)
        self.ref = torch.max
        self.amplitude_to_db = AmplitudeToDB(
            top_db=top_db, stype="power" if power == 2.0 else "magnitude"
        )

    def __call__(self, audio: torch.Tensor, **batch) -> torch.Tensor:
        return self.amplitude_to_db(self.melspec_transform(audio.squeeze(0)))
