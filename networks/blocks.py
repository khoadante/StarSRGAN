import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device("cuda"))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.noise = GaussianNoise()
        self.conv0 = nn.Conv2d(channels, growth_channels, (1, 1), (1, 1), bias=False)

        self.conv1 = nn.Conv2d(
            channels + growth_channels * 0,
            channels + growth_channels * 0,
            (3, 3),
            (1, 1),
            (1, 1),
            groups=channels + growth_channels * 0,
        )
        self.conv1_ = nn.Conv2d(channels + growth_channels * 0, growth_channels, (1, 1))

        self.conv2 = nn.Conv2d(
            channels + growth_channels * 1,
            channels + growth_channels * 1,
            (3, 3),
            (1, 1),
            (1, 1),
            groups=channels + growth_channels * 1,
        )
        self.conv2_ = nn.Conv2d(channels + growth_channels * 1, growth_channels, (1, 1))

        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2,
            channels + growth_channels * 2,
            (3, 3),
            (1, 1),
            (1, 1),
            groups=channels + growth_channels * 2,
        )
        self.conv3_ = nn.Conv2d(channels + growth_channels * 2, growth_channels, (1, 1))

        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3,
            channels + growth_channels * 3,
            (3, 3),
            (1, 1),
            (1, 1),
            groups=channels + growth_channels * 3,
        )
        self.conv4_ = nn.Conv2d(channels + growth_channels * 3, growth_channels, (1, 1))

        self.conv5 = nn.Conv2d(
            channels + growth_channels * 4,
            channels + growth_channels * 4,
            (3, 3),
            (1, 1),
            (1, 1),
            groups=channels + growth_channels * 4,
        )
        self.conv5_ = nn.Conv2d(channels + growth_channels * 4, channels, (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(
            self.conv1_(self.conv1(x)),
        )
        out2 = self.leaky_relu(
            self.conv2_(self.conv2(torch.cat([x, out1], 1))),
        )
        out2 = out2 + self.conv0(x)
        out3 = self.leaky_relu(
            self.conv3_(self.conv3(torch.cat([x, out1, out2], 1))),
        )
        out4 = self.leaky_relu(
            self.conv4_(self.conv4(torch.cat([x, out1, out2, out3], 1))),
        )
        out4 = out4 + out2
        out5 = self.identity(
            self.conv5_(
                self.conv5(torch.cat([x, out1, out2, out3, out4], 1)),
            )
        )
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return self.noise(out)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out
