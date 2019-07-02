import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import math


class Discriminator(nn.Module):
    def __init__(self, img_shape, cat_dim, cont_dim):
        super(Discriminator, self).__init__()

        self.img_shape = tuple(img_shape) # img_shape = (C, H, W) and H is equal to W

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = math.ceil(self.img_shape[1] / (2 ** 4))

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, cat_dim))
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, cont_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code
