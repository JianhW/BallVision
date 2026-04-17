""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    warmup_method = getattr(cfg.SOLVER, 'WARMUP_METHOD', 'cosine')

    if warmup_method == 'cosine':
        # Cosine annealing with warmup
        lr_min = 0.002 * cfg.SOLVER.BASE_LR
        warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
        warmup_t = cfg.SOLVER.WARMUP_EPOCHS
        noise_range = None

        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )
    elif warmup_method == 'step':
        # Step decay: reduce LR by factor every N epochs
        lr_scheduler = StepLR(
            optimizer,
            step_size=30,  # Reduce LR every 30 epochs
            gamma=0.1,     # Reduce by factor of 0.1
        )
    elif warmup_method == 'multistep':
        # Multi-step decay: reduce LR at specific epochs
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=[60, 90],  # Reduce LR at epoch 60 and 90
            gamma=0.1,
        )
    else:
        # Default to cosine
        lr_min = 0.002 * cfg.SOLVER.BASE_LR
        warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
        warmup_t = cfg.SOLVER.WARMUP_EPOCHS
        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
            )

    return lr_scheduler
