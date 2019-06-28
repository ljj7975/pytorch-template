import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel


class Generator(nn.Module):
    def __init__(self, input_size, output_size, img_shape):
        super(Generator, self).__init__()
        self.img_shape = tuple(img_shape)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    # forward method
    def forward(self, input):
        img = self.model(input)
        x = img.view(img.size(0), *self.img_shape)
        return x
