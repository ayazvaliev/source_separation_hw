from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from src.metrics.base_metric import BaseMetric


class SISDR(BaseMetric):
    def __init__(self, name=None):
        super().__init__(name)
        self.metric = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    def __call__(self, audio_concat, logits, **batch):
        return -self.metric(logits, audio_concat)