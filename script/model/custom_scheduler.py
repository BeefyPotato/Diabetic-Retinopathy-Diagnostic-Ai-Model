# custom_scheduler.py

import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingScheduler(_LRScheduler):
    def __init__(
        self, 
        optimizer, 
        warmup_steps, 
        total_steps, 
        min_lrs, 
        last_epoch=-1
    ):
        """
        Initializes the combined Warmup and Cosine Annealing Scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for the warmup phase.
            total_steps (int): Total number of training steps.
            min_lrs (List[float]): Minimum learning rates for each parameter group.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lrs = min_lrs
        self.global_step = 0

        if len(self.min_lrs) != len(optimizer.param_groups):
            raise ValueError(
                "Length of min_lrs must match number of optimizer.param_groups"
            )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1  # _LRScheduler uses last_epoch as the number of completed steps

        lrs = []
        for idx, (base_lr, min_lr) in enumerate(zip(self.base_lrs, self.min_lrs)):
            if current_step < self.warmup_steps:
                # Warmup Phase: Linear increase
                lr = base_lr * (current_step / self.warmup_steps)
            elif current_step <= self.total_steps:
                # Cosine Annealing Phase
                cosine_steps = self.total_steps - self.warmup_steps
                cosine_step = current_step - self.warmup_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_step / cosine_steps))
                lr = min_lr + (base_lr - min_lr) * cosine_decay
            else:
                # After Total Steps: Maintain min_lr
                lr = min_lr

            lrs.append(lr)
        
        # Logging for verification
        if self.global_step % 100 == 99:
            for idx, lr in enumerate(lrs):
                print(f"Step {current_step}: Param Group {idx} LR: {lr}")
        self.global_step += 1
        return lrs
