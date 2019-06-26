import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, generator, discriminator, config, data_loader,
                 valid_data_loader=None):
        super().__init__(generator, discriminator, config)
        self.config = config
        self.data_loader = data_loader

        if self.config['trainer'].get('len_epoch', None):
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        else:
            # epoch-based training
            self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(len(data_loader)/5) if len(data_loader) > 5 else 1

        self.z_size = self.generator['config']['arch']['args']['input_size']
        self.output_size = self.generator['config']['arch']['args']['output_size']

    def _eval_metrics(self, metrics, writer, output, target):
        acc_metrics = np.zeros(len(metrics))
        for i, metric in enumerate(metrics):
            acc_metrics[i] += metric(output, target)
            writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_discriminator(self, epoch, batch_idx, real_data, fake_data):
        self.generator['model'].eval()
        self.discriminator['model'].train()

        self.discriminator['optimizer'].zero_grad()

        real_target = Variable(torch.FloatTensor(real_data.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
        fake_target = Variable(torch.FloatTensor(fake_data.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

        data = torch.cat((real_data, fake_data), 0)
        target = torch.cat((real_target, fake_target), 0)

        output = self.discriminator['model'](data)
        loss = self.discriminator['loss_fn'](output, target)

        loss.backward()
        self.discriminator['optimizer'].step()

        self.discriminator['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.discriminator['writer'].add_scalar('loss', loss.item())

        metrics = self._eval_metrics(self.discriminator['metric_fns'], self.discriminator['writer'], output, target.long())

        return loss.item(), metrics

    def _train_generator(self, epoch, batch_idx, data):
        self.generator['model'].train()
        self.discriminator['model'].eval()

        self.generator['optimizer'].zero_grad()

        # generate samples which can fool discriminator
        target = Variable(torch.FloatTensor(data.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)

        # Sample noise as generator input
        z = Variable(torch.randn(data.size(0), self.z_size))

        # Generate a batch of images
        gen_samples = self.generator['model'](z)
        dis_output = self.discriminator['model'](gen_samples)

        # Loss measures generator's ability to fool the discriminator
        loss = self.generator['loss_fn'](dis_output, target)

        loss.backward()
        self.generator['optimizer'].step()
        gen_samples = gen_samples.detach()

        self.generator['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.generator['writer'].add_scalar('loss', loss.item())
        metrics = self._eval_metrics(self.generator['metric_fns'], self.generator['writer'], dis_output, target.long())

        return loss.item(), metrics, gen_samples

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

            The metrics in log must have the key 'metrics'.
        """

        self.generator['model'].train()
        self.discriminator['model'].train()

        assert self.generator['config']['arch']['args']['output_size'] == \
            self.discriminator['config']['arch']['args']['input_size']

        total_gen_loss = 0
        total_gen_metrics = np.zeros(len(self.generator['metric_fns']))

        total_dis_loss = 0
        total_dis_metrics = np.zeros(len(self.discriminator['metric_fns']))

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            gen_loss, gen_metrics, gen_samples = self._train_generator(epoch, batch_idx, data)

            total_gen_loss += gen_loss
            total_gen_metrics += gen_metrics

            dis_loss, dis_metrics = self._train_discriminator(epoch, batch_idx, data, gen_samples)

            total_dis_loss += dis_loss
            total_dis_metrics += dis_metrics

            if batch_idx % self.log_step == 0:
                self.logger.debug(('Train Epoch: {} {} \n\t'
                    + 'generator loss\t\t: {:.6f} \n\t'
                    + 'discriminator loss\t: {:.6f}').format(
                    epoch,
                    self._progress(batch_idx),
                    gen_loss,
                    dis_loss))
                self.generator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                self.discriminator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = {
            'generator': {
                'loss': total_gen_loss / self.len_epoch,
                'metrics': (total_gen_metrics / self.len_epoch).tolist(),
            },
            'discriminator': {
                'loss': total_dis_loss / self.len_epoch,
                'metrics': (total_dis_metrics / self.len_epoch).tolist()
            }
        }

        # TODO :: implement validation steps
        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log.update(val_log)

        if self.generator['lr_scheduler'] is not None:
            self.generator['lr_scheduler'].step()

        if self.discriminator['lr_scheduler'] is not None:
            self.discriminator['lr_scheduler'].step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
