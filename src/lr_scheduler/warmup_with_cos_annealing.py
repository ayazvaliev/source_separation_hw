import torch.optim as optim


class WarmupWithCosAnnealing:
    def __init__(self, optimizer, total_steps, warmup_ratio, start_factor=None):
        warmup_steps = int(total_steps * warmup_ratio)

        if warmup_steps == 0:
            self._lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            return

        assert start_factor is not None, "start_factor must be provided when warmup_steps > 0"

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, total_iters=warmup_steps
        )

        cos_annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(total_steps - warmup_steps)
        )

        self._lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cos_annealing_scheduler],
            milestones=[warmup_steps],
        )

    def __getattr__(self, name):
        return getattr(self._lr_scheduler, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        if hasattr(self, "_lr_scheduler"):
            setattr(self._lr_scheduler, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith("_"):
            super().__delattr__(name)
            return

        if hasattr(self, "_lr_scheduler"):
            delattr(self._lr_scheduler, name)
        else:
            super().__delattr__(name)
