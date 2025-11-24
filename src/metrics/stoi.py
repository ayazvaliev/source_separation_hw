import torch
from asteroid.metrics import get_metrics

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, sample_rate, name=None):
        super().__init__(name)
        self.sample_rate = sample_rate

    def __call__(self, audio_mix, audio_concat, logits, **batch):
        batch_size = audio_mix.size(0)
        audio_mix = audio_mix.cpu().numpy()
        target_audio = audio_concat.cpu().numpy()
        logits = logits.cpu().numpy()
        accum = 0

        for mix, clean, log in zip(audio_mix, target_audio, logits):
            metrices = get_metrics(
                mix=mix,
                clean=clean,
                estimate=log,
                sample_rate=self.sample_rate,
                compute_permutation=True,
                average=True,
                metrics_list=["stoi"],
            )
            accum += metrices["stoi"] / batch_size

        return accum
