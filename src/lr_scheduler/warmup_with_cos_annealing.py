import torch.optim as optim

class WarmupWithCosAnnealing:
    def __init__(self, optimizer, total_steps, warmup_ratio, start_factor=None):
        warmup_steps = int(total_steps * warmup_ratio)
        if warmup_steps == 0:
            self._lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps
            )
            return
        assert start_factor is not None
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=warmup_steps
        )
        cos_annealing_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps
        )
        self._lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cos_annealing_scheduler],
            milestones=[warmup_scheduler]
        )

    def __getattr__(self, name):
        return getattr(self._lr_scheduler, name)

    def __setattr__(self, name, value):
        if name == "_lr_scheduler":
            pass
        else:
            setattr(self._lr_scheduler, name, value)
    
    def __delattr__(self, name):
        if name == "_lr_scheduler":
            super().__delattr__(name)
        else:
            delattr(self._lr_scheduler, name)