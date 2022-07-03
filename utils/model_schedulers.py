from typing import List

from torch import optim
from torch.optim import lr_scheduler

import config


def define_pegasusnet_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(
        optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma
    )

    return scheduler


def define_pegasusgan_scheduler(
    d_optimizer: optim.Adam, g_optimizer: optim.Adam
) -> List[lr_scheduler.MultiStepLR]:
    d_scheduler = lr_scheduler.MultiStepLR(
        d_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma
    )
    g_scheduler = lr_scheduler.MultiStepLR(
        g_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma
    )

    return d_scheduler, g_scheduler
