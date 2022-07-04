from typing import Union

from torch import nn

from networks.losses import ContentLoss
import config


def define_starsrnet_loss() -> nn.L1Loss:
    pixel_criterion = nn.L1Loss()
    pixel_criterion = pixel_criterion.to(device=config.device)

    return pixel_criterion


def define_starsrgan_loss() -> Union[nn.L1Loss, ContentLoss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.L1Loss()
    content_criterion = ContentLoss(
        config.feature_model_extractor_nodes,
        config.feature_model_normalize_mean,
        config.feature_model_normalize_std,
    )
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=config.device)
    content_criterion = content_criterion.to(device=config.device)
    adversarial_criterion = adversarial_criterion.to(device=config.device)

    return pixel_criterion, content_criterion, adversarial_criterion
