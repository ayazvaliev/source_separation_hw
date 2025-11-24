import random

import torch
import torch.nn as nn
import torchaudio.transforms as ta_transforms


class SpeedPerturb(nn.Module):
    """
    Resample-based speed perturbation as an nn.Module.

    Behavior:
      - For each example in the batch, with probability p it picks a random factor in [min_speed, max_speed]
        and resamples to new_sr = round(sr * factor) then resamples back to sr.
      - Caches torchaudio.transforms.Resample objects keyed by (orig_sr, new_sr, device, dtype)
        so repeated forward calls re-use kernels.
      - Ensures output has same number of frames as input by center-trim or random-pad.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_speed: float = 0.92,
        max_speed: float = 1.08,
        p: float = 0.25,
    ):
        super().__init__()
        assert min_speed > 0 and max_speed > 0 and min_speed <= max_speed
        self.sr = int(sample_rate)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.p = float(p)
        # cache of resamplers: dict[(orig_sr,new_sr,device,dtype)] = (resample_to_new, resample_back)
        self._resampler_cache = {}

    def _get_resamplers(self, orig_sr: int, new_sr: int, device, dtype):
        key = (orig_sr, new_sr, device, str(dtype))
        if key in self._resampler_cache:
            return self._resampler_cache[key]
        # create and store
        r1 = ta_transforms.Resample(orig_freq=orig_sr, new_freq=new_sr).to(
            device=device, dtype=dtype
        )
        r2 = ta_transforms.Resample(orig_freq=new_sr, new_freq=orig_sr).to(
            device=device, dtype=dtype
        )
        self._resampler_cache[key] = (r1, r2)
        return r1, r2

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: (B, C, T)
        returns: (B, C, T)
        """
        if not self.training or random.random() > self.p:
            # no-op (maintains exact input tensor reference semantics)
            return batch

        if batch.dim() != 3:
            raise ValueError(
                "SpeedPerturb expects input shape (B, C, T). Got: {}".format(batch.shape)
            )

        B, C, T = batch.shape
        device = batch.device
        dtype = batch.dtype

        out = torch.empty_like(batch, device=device, dtype=dtype)

        # For each example, do independent perturbation
        for i in range(B):
            x = batch[i]  # (C, T)
            factor = random.uniform(self.min_speed, self.max_speed)
            # keep near-1.0 as identity to avoid unnecessary resampling work
            if abs(factor - 1.0) < 1e-6:
                out[i] = x
                continue
            new_sr = max(1, int(round(self.sr * factor)))
            if new_sr == self.sr:
                out[i] = x
                continue
            r_to_new, r_back = self._get_resamplers(self.sr, new_sr, device, dtype)
            # resample to new_sr then back
            y = r_to_new(x)  # shape (C, T_new)
            y = r_back(y)  # shape (C, T') approximately T
            # ensure length = T by trim/pad (centered trim, random pad placement)
            t_out = y.shape[-1]
            if t_out == T:
                out[i] = y
            elif t_out > T:
                # center-crop to T to avoid bias
                start = (t_out - T) // 2
                out[i] = y[..., start : start + T]
            else:
                # pad; random left padding to distribute padding locations
                pad = T - t_out
                left = random.randint(0, pad)
                right = pad - left
                out[i] = torch.nn.functional.pad(y, (left, right))
        return out
