"""
Discriminators to be used in GAN systems.
"""
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, img_size=512):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [
                nn.Conv2d(in_feat, out_feat, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),  # a linear layer
        )

    def forward(self, input):
        out = self.model(input)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
