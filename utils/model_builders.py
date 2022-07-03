from typing import List

from torch import nn

from networks.models import EMA, Generator, Discriminator
import config


def build_pegasusnet_model() -> List[nn.Module]:
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor)
    model = model.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_model = EMA(model, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device)
    ema_model.register()

    return model, ema_model


def build_pegasusgan_model() -> List[nn.Module]:
    discriminator = Discriminator()
    generator = Generator(
        config.in_channels, config.out_channels, config.upscale_factor
    )

    # Transfer to CUDA
    discriminator = discriminator.to(device=config.device)
    generator = generator.to(device=config.device)

    # Create an Exponential Moving Average Model
    ema_model = EMA(generator, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device)
    ema_model.register()

    return discriminator, generator, ema_model
