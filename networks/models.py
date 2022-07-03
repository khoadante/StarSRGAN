import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from networks.blocks import ResidualResidualDenseBlock
import config


class EMA(nn.Module):
    def __init__(self, model: nn.Module, weight_decay: float) -> None:
        super(EMA, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.shadow = {}
        self.backup = {}

    def register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.weight_decay
                ) * param.data + self.weight_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        self.down_block1 = nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False, groups=64)
        self.down_block1_ = spectral_norm(nn.Conv2d(64, 128, (1, 1), bias=False))

        self.down_block2 = nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False, groups=128)
        self.down_block2_ = spectral_norm(nn.Conv2d(128, 256, (1, 1), bias=False))

        self.down_block3 = nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False, groups=256)
        self.down_block3_ = spectral_norm(nn.Conv2d(256, 512, (1, 1), bias=False))

        self.up_block1 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False, groups=512)
        self.up_block1_ = spectral_norm(nn.Conv2d(512, 256, (1, 1), bias=False))

        self.up_block2 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=False, groups=256)
        self.up_block2_ = spectral_norm(nn.Conv2d(256, 128, (1, 1), bias=False))

        self.up_block3 = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), bias=False, groups=128)
        self.up_block3_ = spectral_norm(nn.Conv2d(128, 64, (1, 1), bias=False))

        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False, groups=64)
        self.conv2_ = spectral_norm(nn.Conv2d(64, 64, (1, 1), bias=False))

        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False, groups=64)
        self.conv3_ = spectral_norm(nn.Conv2d(64, 64, (1, 1), bias=False))

        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.conv4_ = nn.Conv2d(64, 1, (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)

        # Down-sampling
        down1 = self.leaky_relu(self.down_block1_(self.down_block1(out1)))
        down2 = self.leaky_relu(self.down_block2_(self.down_block2(down1)))
        down3 = self.leaky_relu(self.down_block3_(self.down_block3(down2)))

        # Up-sampling
        down3 = F.interpolate(
            down3, scale_factor=2, mode="bilinear", align_corners=False
        )
        up1 = self.leaky_relu(self.up_block1_(self.up_block1(down3)))

        up1 = torch.add(up1, down2)
        up1 = F.interpolate(up1, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = self.leaky_relu(self.up_block2_(self.up_block2(up1)))

        up2 = torch.add(up2, down1)
        up2 = F.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = self.leaky_relu(self.up_block3_(self.up_block3(up2)))

        up3 = torch.add(up3, out1)

        out = self.leaky_relu(self.conv2_(self.conv2(up3)))
        out = self.leaky_relu(self.conv3_(self.conv3(out)))
        out = self.conv4_(self.conv4(out))

        return out


class Generator(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, upscale_factor: int = 4
    ) -> None:
        super(Generator, self).__init__()
        if upscale_factor == 2:
            in_channels *= 4
            downscale_factor = 2
        elif upscale_factor == 1:
            in_channels *= 16
            downscale_factor = 4
        else:
            in_channels *= 1
            downscale_factor = 1

        # Down-sampling layer
        self.downsampling = nn.PixelUnshuffle(downscale_factor)

        # The first layer of convolutional layer
        self.conv1 = nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1), groups=in_channels)
        self.conv1_ = nn.Conv2d(in_channels, 64, (1, 1))

        # Feature extraction backbone network
        trunk = []
        for _ in range(23):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.conv2_ = nn.Conv2d(64, 64, (1, 1))

        # Upsampling convolutional layer
        self.upsampling1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.upsampling1_ = nn.Conv2d(64, 64, (1, 1))

        self.upsampling2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.upsampling2_ = nn.Conv2d(64, 64, (1, 1))

        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.conv3_ = nn.Conv2d(64, 64, (1, 1))

        # Output layer
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), groups=64)
        self.conv4_ = nn.Conv2d(64, out_channels, (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # If upscale_factor not equal 4, must use nn.PixelUnshuffle() ops
        out = self.downsampling(x)

        out1 = self.conv1_(self.conv1(out))
        out = self.trunk(out1)
        out2 = self.conv2_(self.conv2(out))
        out = torch.add(out1, out2)

        out = self.leaky_relu(self.upsampling1_(self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))))
        out = self.leaky_relu(self.upsampling2_(self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))))

        out = self.leaky_relu(self.conv3_(self.conv3(out)))
        out = self.conv4_(self.conv4(out))

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
