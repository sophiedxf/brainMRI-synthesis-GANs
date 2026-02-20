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


def _num_downsample_blocks(image_size: int) -> int:
    if image_size not in VALID_IMAGE_SIZES:
        raise ValueError(f"image_size must be one of {sorted(VALID_IMAGE_SIZES)}")
    return int(math.log2(image_size) - 2)  # down to 4x4


class WGANGPCritic(nn.Module):
    """
    Critic outputs real-valued score (no sigmoid).
    Uses InstanceNorm (OK for WGAN-GP) or you can remove norms.
    """

    def __init__(self, image_size=256, ndf=64, in_channels=1):
        super().__init__()
        n_blocks = _num_downsample_blocks(image_size)

        layers = []

        # first downsample
        ch = ndf
        layers += [
            nn.Conv2d(in_channels, ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # more downsamples
        for _ in range(n_blocks - 1):
            next_ch = ch * 2
            layers += [
                nn.Conv2d(ch, next_ch, 4, 2, 1),
                nn.InstanceNorm2d(next_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch

        # final 4x4 -> score
        layers += [nn.Conv2d(ch, 1, 4, 1, 0)]

        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x).view(-1)


def gradient_penalty(critic, real, fake, device, lambda_gp=10.0):
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, 1, 1, device=device)
    x_hat = alpha * real + (1.0 - alpha) * fake
    x_hat.requires_grad_(True)

    scores = critic(x_hat)
    grad_outputs = torch.ones_like(scores, device=device)

    grads = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(bsz, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return lambda_gp * gp
