from src.metrics.base_metric import BaseMetric
from asteroid.metrics import get_metrics
import torch


class SDRi(BaseMetric):
    def __init__(self, sample_rate, name=None):
            super().__init__(name)
            self.sample_rate = sample_rate

    def __call__(self, audio_mix, audio_s1, audio_s2, logits, **batch):
        audio_mix = audio_mix.cpu().numpy()
        target_audio = torch.cat([audio_s1, audio_s2], dim=1).cpu().numpy()
        logits = logits.cpu().numpy()
        accum = 0
        batch_size = audio_mix.size(0)

        for mix, clean, log in zip(audio_mix, target_audio, logits):
            metrices = get_metrics(mix=mix,
                                   clean=clean,
                                   estimate=log,
                                   sample_rate=self.sample_rate,
                                   compute_permutation=True,
                                   average=True,
                                   metrics_list=["sdr"])
            accum += (metrices['sdr'] - metrices['input_sdr']) / batch_size
        

        return accum
