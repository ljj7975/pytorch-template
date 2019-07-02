import os
import torch
from abc import abstractmethod
import numpy as np
from torch.autograd import Variable
from utils import to_categorical


class BaseEvaluator:
    """
    Base class for all evaluators
    """
    def __init__(self, config, logger, generator, discriminator, encoder):
        self.config = config
        self.logger = logger

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = generator
        self.generator['model'] = self.generator['model'].to(self.device)

        self.discriminator = discriminator
        self.discriminator['model'] = self.discriminator['model'].to(self.device)

        self.encoder = encoder

        self.generator['config'] = config['generator']
        self.discriminator['config'] = config['discriminator']
        self.encoder['config'] = config['encoder']

        self.lambda_cat = self.encoder['config']['lambda_cat']
        self.lambda_con = self.encoder['config']['lambda_con']

        self.latent_dim = config['generator']['arch']['args']['latent_dim']
        self.cat_dim = config['generator']['arch']['args']['cat_dim']
        self.cont_dim = config['generator']['arch']['args']['cont_dim']

    @abstractmethod
    def evaluate(self):
        """
        Evaluation logic
        """
        raise NotImplementedError

    def _generate_random_input(self, batch_size):
        z = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))).float().to(self.device)
        label_input = to_categorical(np.random.randint(0, self.cat_dim, batch_size), num_columns=self.cat_dim).to(self.device)
        code_input = Variable(torch.Tensor(np.random.uniform(-1, 1, (batch_size, self.cont_dim)))).float().to(self.device)

        z.requires_grad = False
        label_input.requires_grad = False
        code_input.requires_grad = False

        return z, label_input, code_input
