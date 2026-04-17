import math
import torch
import torch.nn as nn

from config import VALID_IMAGE_SIZES


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def _num_upsample_blocks(image_size: int) -> int:
    """
    From 4x4 -> image_size via doubling.
    64: 4 blocks (4->8->16->32->64)
    128: 5 blocks
    256: 6 blocks
    """
    if image_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"image_size must be one of {sorted(VALID_IMAGE_SIZES)}")
    return int(math.log2(image_size) - 2)


class DCGANGenerator(nn.Module):
    """
    z: (B, z_dim, 1, 1) -> (B, out_channels, image_size, image_size)
    """

    def __init__(self, image_size=256, z_dim=128, ngf=64, out_channels=1):
        super().__init__()
        n_blocks = _num_upsample_blocks(image_size)

        layers = []

        # 1x1 -> 4x4
        ch = ngf * (2 ** n_blocks)  # largest channel width
        layers += [
            nn.ConvTranspose2d(z_dim, ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
        ]

        # 4x4 -> ... -> image_size
        for _ in range(n_blocks):
            next_ch = ch // 2
            layers += [
                nn.ConvTranspose2d(ch, next_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(True),
            ]
            ch = next_ch

        # final: ensure correct output channels (after loop we are at ngf)
        layers += [
            nn.Conv2d(ch, out_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """
    x: (B, in_channels, image_size, image_size) -> (B,1) prob real
    """

    def __init__(self, image_size=256, ndf=64, in_channels=1):
        super().__init__()
        n_blocks = _num_upsample_blocks(image_size)  # same count for downsampling to 4x4

        layers = []

        # image_size -> image_size/2
        ch = ndf
        layers += [
            nn.Conv2d(in_channels, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # progressively downsample until 4x4
        for _ in range(n_blocks - 1):
            next_ch = ch * 2
            layers += [
                nn.Conv2d(ch, next_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch

        # now should be at 4x4, map to scalar
        layers += [
            nn.Conv2d(ch, 1, 4, 1, 0, bias=False),
        ]

        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x).view(-1, 1)
