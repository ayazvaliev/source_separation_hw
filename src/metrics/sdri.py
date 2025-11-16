from src.metrics.base_metric import BaseMetric
import torch
from torchmetrics.functional.audio import (
     permutation_invariant_training,
     scale_invariant_signal_distortion_ratio,
     signal_distortion_ratio
)
from src.metrics.metric_utils import gather_by_perm


class SISNRi(BaseMetric):
    def __init__(self, name=None):
            super().__init__(name)

    def __call__(self, audio_mix, audio_s1, audio_s2, logits, **batch):
        B, S, T = logits.shape
        audio_mix = audio_mix.expand(-1, S, -1)
        clean = torch.concat([audio_s1, audio_s2], dim=1)

        _, best_permut = permutation_invariant_training(
             preds=logits, target=clean,
             metric_func=scale_invariant_signal_distortion_ratio,
             eval_func='max',
             zero_mean=True
        )
        logits = gather_by_perm(logits, best_permut)

        sdr_est = signal_distortion_ratio(logits, clean, zero_mean=True)
        sdr_mix = signal_distortion_ratio(audio_mix, clean, zero_mean=True)
        sdri = sdr_est - sdr_mix

        return sdri.mean().item()
