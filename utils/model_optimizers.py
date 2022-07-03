from typing import List

from torch import nn
from torch import optim

import config


def define_pegasusnet_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_pegasusgan_optimizer(
    discriminator: nn.Module, generator: nn.Module
) -> List[optim.Adam]:
    d_optimizer = optim.Adam(
        discriminator.parameters(), config.model_lr, config.model_betas
    )
    g_optimizer = optim.Adam(
        generator.parameters(), config.model_lr, config.model_betas
    )

    return d_optimizer, g_optimizer
