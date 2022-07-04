from typing import List

from torch import nn
from torch import optim

import config


def define_pegasusnet_optimizer(model) -> optim.AdamW:
    optimizer = optim.AdamW(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_pegasusgan_optimizer(
    discriminator: nn.Module, generator: nn.Module
) -> List[optim.AdamW]:
    d_optimizer = optim.AdamW(
        discriminator.parameters(), config.model_lr, config.model_betas
    )
    g_optimizer = optim.AdamW(
        generator.parameters(), config.model_lr, config.model_betas
    )

    return d_optimizer, g_optimizer
