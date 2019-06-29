import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from base import BaseTrainer
from utils import inf_loop

def evaluate(generator, discriminator, config, data_loader):

    def evaluate_generator(batch_idx, generator, discriminator, z_size, data):
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        generator['model'] = generator['model'].to(device)
        discriminator['model'] = discriminator['model'].to(device)

        generator['model'].eval()
        discriminator['model'].eval()

        # Sample noise as generator input
        z = Variable(torch.randn(data.size(0), z_size)).to(device)

        # Generate a batch of images
        gen_samples = generator['model'](z)
        dis_output = discriminator['model'](gen_samples)

        loss = -torch.mean(dis_output)

        gen_samples = gen_samples.detach()

        return loss.item(), gen_samples

    def evaluate_discriminator(batch_idx, generator, discriminator, z_size, data):
        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        # fake_data = fake_data.to(device)
        generator['model'] = generator['model'].to(device)
        discriminator['model'] = discriminator['model'].to(device)

        generator['model'].eval()
        discriminator['model'].eval()

        # Sample noise as generator input
        z = Variable(torch.randn(data.size(0), z_size)).to(device)

        # Generate a batch of images
        gen_samples = generator['model'](z)
        fake_output = discriminator['model'](gen_samples)

        real_output = discriminator['model'](data)

        loss = -(torch.mean(real_output) - torch.mean(fake_output))\

        return loss.item()

    assert config['generator']['arch']['args']['output_size'] == \
        config['discriminator']['arch']['args']['input_size']

    z_size = config['generator']['arch']['args']['input_size']

    total_gen_loss = 0
    total_dis_loss = 0

    for batch_idx, (data, _) in enumerate(data_loader):

        dis_loss = evaluate_discriminator(batch_idx, generator, discriminator, z_size, data)

        total_dis_loss += dis_loss

        gen_loss, gen_samples = evaluate_generator(batch_idx, generator, discriminator, z_size, data)

        total_gen_loss += gen_loss

    num_batch = len(data_loader)

    log = {
        'generator': {
            'loss': total_gen_loss / num_batch
        },
        'discriminator': {
            'loss': total_dis_loss / num_batch
        }
    }

    return log

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, config, logger, generator, discriminator, gif_generator, data_loader,
                 valid_data_loader=None):
        super().__init__(config, logger, generator, discriminator, gif_generator)
        self.config = config
        self.data_loader = data_loader

        if self.config['trainer'].get('len_epoch', None):
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        else:
            # epoch-based training
            self.len_epoch = len(self.data_loader)

        self.clip_value = discriminator["config"]["clip_value"]

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(len(data_loader)/5) if len(data_loader) > 5 else 1

        self.z_size = self.generator['config']['arch']['args']['input_size']
        self.output_size = self.generator['config']['arch']['args']['output_size']

    def _train_discriminator(self, epoch, batch_idx, data):
        self.generator['model'].eval()
        self.discriminator['model'].train()

        self.discriminator['optimizer'].zero_grad()

        # Sample noise as generator input
        z = Variable(torch.randn(data.size(0), self.z_size)).to(self.device)

        # Generate a batch of images
        gen_samples = self.generator['model'](z)
        fake_output = self.discriminator['model'](gen_samples)

        real_output = self.discriminator['model'](data)

        loss = -(torch.mean(real_output) - torch.mean(fake_output))

        loss.backward()
        self.discriminator['optimizer'].step()

        # Clip weights of discriminator
        for p in self.discriminator['model'].parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

        self.discriminator['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.discriminator['writer'].add_scalar('loss', loss.item())

        return loss.item()

    def _train_generator(self, epoch, batch_idx, data):
        self.generator['model'].train()
        self.discriminator['model'].eval()

        self.generator['optimizer'].zero_grad()

        # generate samples which can fool discriminator
        target = Variable(torch.FloatTensor(data.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)

        # Sample noise as generator input
        z = Variable(torch.randn(data.size(0), self.z_size)).to(self.device)

        # Generate a batch of images
        gen_samples = self.generator['model'](z)
        dis_output = self.discriminator['model'](gen_samples)

        loss = -torch.mean(dis_output)

        loss.backward()
        self.generator['optimizer'].step()
        gen_samples = gen_samples.detach()

        self.generator['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.generator['writer'].add_scalar('loss', loss.item())

        return loss.item(), gen_samples

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """

        assert self.generator['config']['arch']['args']['output_size'] == \
            self.discriminator['config']['arch']['args']['input_size']

        total_gen_loss = 0
        total_dis_loss = 0

        for batch_idx, (data, _) in enumerate(self.data_loader):
            data = data.to(self.device)

            dis_loss = self._train_discriminator(epoch, batch_idx, data)

            total_dis_loss += dis_loss
            # total_dis_metrics += dis_metrics

            if batch_idx % self.log_step == 0:
                log_msg = ('Train Epoch: {} {} \n\t'
                    + 'discriminator loss\t: {:.6f}\n').format(
                    epoch,
                    self._progress(batch_idx),
                    dis_loss)
                self.discriminator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                self.generator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if epoch % self.n_critic == 0:
                gen_loss, gen_metrics = self._train_generator(epoch, batch_idx, data)

                total_gen_loss += gen_loss

                if batch_idx % self.log_step == 0:
                    log_msg += ('\tgenerator loss\t: {:.6f}').format(gen_loss)
                    self.generator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx % self.log_step == 0:
                self.logger.debug(log_msg)

            if batch_idx == self.len_epoch:
                break

        log = {
            'generator': {
                'loss': total_gen_loss / self.len_epoch,
            },
            'discriminator': {
                'loss': total_dis_loss / self.len_epoch,
            }
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log['generator'].update(val_log['generator'])
            log['discriminator'].update(val_log['discriminator'])

        if self.generator['lr_scheduler'] is not None:
            self.generator['lr_scheduler'].step()

        if self.discriminator['lr_scheduler'] is not None:
            self.discriminator['lr_scheduler'].step()

        return log

    def _valid_epoch(self, epoch):
        val_log = evaluate(self.generator, self.discriminator, self.config, self.valid_data_loader)

        gen_val_log = {}
        for key, value in val_log['generator'].items():
            gen_val_log['val_'+key] = val_log['generator'][key]

        self.generator['writer'].set_step(epoch * len(self.valid_data_loader), 'valid')
        for key, value in gen_val_log.items():
            self.generator['writer'].add_scalar('val_'+key, value)

        dis_val_log = {}
        for key, value in val_log['discriminator'].items():
            dis_val_log['val_'+key] = val_log['discriminator'][key]

        self.discriminator['writer'].set_step(epoch * len(self.valid_data_loader), 'valid')
        for key, value in dis_val_log.items():
            self.discriminator['writer'].add_scalar('val_'+key, value)

        val_log['generator'] = gen_val_log
        val_log['discriminator'] = dis_val_log

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator['model'].named_parameters():
            self.generator['writer'].add_histogram(name, p, bins='auto')

        # add histogram of model parameters to the tensorboard
        for name, p in self.discriminator['model'].named_parameters():
            self.discriminator['writer'].add_histogram(name, p, bins='auto')

        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
