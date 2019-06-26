import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel


class Generator(nn.Module):
    def __init__(self, input_size, output_size, img_shape):
        super(Generator, self).__init__()
        self.img_shape = tuple(img_shape)
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_size)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        x = x.view(x.size(0), *self.img_shape)
        return x
