from torch import nn
import torch


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, n_class, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(64, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, audio_mix: torch.Tensor, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        # audio_mix (N, L, C=1)
        return {"logits": self.net(audio_mix)} # (N, L, NUM_CLASSES=2)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
