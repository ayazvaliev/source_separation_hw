import torch
import torch.nn as nn
from asteroid.losses import PITLossWrapper, pairwise_neg_snr


class SNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss= PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
    
    def forward(self, audio_concat, logits, **batch):
        return {"loss": self.loss(logits, audio_concat)}