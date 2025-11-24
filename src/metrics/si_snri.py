from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_noise_ratio,
)

from src.metrics.base_metric import BaseMetric
from src.metrics.metric_utils import gather_by_perm


class SISNRi(BaseMetric):
    def __init__(self, name=None):
        super().__init__(name)

    def __call__(self, audio_mix, audio_concat, logits, **batch):
        B, S, T = logits.shape
        audio_mix = audio_mix.expand(-1, S, -1)
        target_audio = audio_concat

        _, best_permut = permutation_invariant_training(
            preds=logits,
            target=target_audio,
            metric_func=scale_invariant_signal_noise_ratio,
            eval_func="max",
        )
        logits = gather_by_perm(logits, best_permut)

        si_snr_est = scale_invariant_signal_noise_ratio(logits, target_audio)
        si_snr_mix = scale_invariant_signal_noise_ratio(audio_mix, target_audio)
        si_sdri = si_snr_est - si_snr_mix

        return si_sdri.mean().item()
