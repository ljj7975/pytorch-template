import os
import torch
from abc import abstractmethod
import numpy as np
from torch.autograd import Variable
from utils import save_generated_images, to_categorical


class BaseEvaluator:
    """
    Base class for all evaluators
    """
    def __init__(self, config, logger, generator, discriminator, encoder):
        self.config = config
        self.logger = logger

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])

        self.generator = self.initialize_training(generator)
        self.discriminator = self.initialize_training(discriminator)
        self.encoder = encoder

        self.generator['config'] = config['generator']
        self.discriminator['config'] = config['discriminator']
        self.encoder['config'] = config['encoder']

        self.lambda_cat = self.encoder['config']['lambda_cat']
        self.lambda_con = self.encoder['config']['lambda_con']

        self.z_size = config['generator']['arch']['args']['latent_dim']
        self.n_classes = config['generator']['arch']['args']['n_classes']
        self.code_dim = config['generator']['arch']['args']['code_dim']

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    @abstractmethod
    def evaluate(self):
        """
        Evaluation logic
        """
        raise NotImplementedError

    def _generate_random_input(self, batch_size):
        z = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, self.z_size)))).float().to(self.device)
        label_input = to_categorical(np.random.randint(0, self.n_classes, batch_size), num_columns=self.n_classes).to(self.device)
        code_input = Variable(torch.Tensor(np.random.uniform(-1, 1, (batch_size, self.code_dim)))).float().to(self.device)

        z.requires_grad = False
        label_input.requires_grad = False
        code_input.requires_grad = False

        return z, label_input, code_input

    def generate_image(self):
        z, cat_c, cont_c = self._generate_random_input(self.num_batch)

        self.generator['model'].eval()
        samples = self.generator['model'](z, cat_c, cont_c).detach().cpu()
        output_file = str(self.config.output_dir / "sample_images.png")

        img = save_generated_images(samples, output_file)
        self.logger.info("saved generated images at {}".format(output_file))

    def initialize_training(self, player):
        player['model'] = player['model'].to(self.device)
        if len(self.device_ids) > 1:
            player['model'] = torch.nn.DataParallel(player['model'], device_ids=device_ids)

        return player
