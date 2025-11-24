import torch
import torch.nn as nn


def calculate_rms(samples: torch.Tensor):
    return samples.square().mean(dim=-1).sqrt()


def rms_normalize(samples: torch.Tensor):
    rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
    return samples / (rms + 1e-8)


class Mix(nn.Module):
    def __init__(self, min_snr_in_db, max_snr_in_db, shuffle_batch=True):
        super().__init__()
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.shuffle_batch = shuffle_batch
        self.snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
            ),
            high=torch.tensor(self.max_snr_in_db, dtype=torch.float32),
            validate_args=True,
        )

    def forward(self, audio_s1: torch.Tensor, audio_s2: torch.Tensor, **batch):
        batch_size = audio_s1.size(0)
        if self.shuffle_batch:
            audio_s2 = audio_s2[torch.randperm(batch_size, device=audio_s2.device)]
        snr = self.snr_distribution.sample(sample_shape=(batch_size,)).to(audio_s1.device)

        background_samples = rms_normalize(audio_s2)
        background_rms = calculate_rms(audio_s1) / (10 ** (snr.unsqueeze(-1) / 20))
        mixed_samples = audio_s1 + background_rms.unsqueeze(-1) * background_samples

        return mixed_samples, audio_s1, audio_s2
