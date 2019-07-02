import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
nn.functional.interpolate

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, n_classes, code_dim):
        super(Generator, self).__init__()
        self.img_shape = tuple(img_shape) # img_shape = (C, H, W) and H is equal to W
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = self.img_shape[1] // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        # img = img.view(img.size(0), *self.img_shape)
        return img
