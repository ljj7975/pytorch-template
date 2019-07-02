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
    def __init__(self, config, logger, generator, discriminator, encoder, gif_generator, data_loader,
                 valid_data_loader=None):
        super().__init__(config, logger, generator, discriminator, encoder, gif_generator, valid_data_loader)
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

    def _eval_metrics(self, metrics, writer, output, target):
        acc_metrics = np.zeros(len(metrics))
        for i, metric in enumerate(metrics):
            acc_metrics[i] += metric(output, target)
            writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_generator(self, epoch, batch_idx, batch_size):
        self.generator['model'].train()
        self.discriminator['model'].eval()

        self.generator['optimizer'].zero_grad()

        # generate samples which can fool discriminator
        target = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(self.device)

        # Sample noise as generator input
        z, cat_c, cont_c = self._generate_random_input(batch_size)

        # Generate a batch of images
        gen_samples = self.generator['model'](z, cat_c, cont_c)
        dis_output, _, _ = self.discriminator['model'](gen_samples)

        # Loss measures generator's ability to fool the discriminator
        loss = self.generator['loss_fn'](dis_output, target)

        loss.backward()
        self.generator['optimizer'].step()
        gen_samples = gen_samples.detach()

        self.generator['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.generator['writer'].add_scalar('loss', loss.item())

        metrics = self._eval_metrics(self.generator['metric_fns'], self.generator['writer'], dis_output, target)
        return loss.item(), metrics, gen_samples

    def _train_discriminator(self, epoch, batch_idx, real_data, fake_data):
        self.generator['model'].eval()
        self.discriminator['model'].train()

        self.discriminator['optimizer'].zero_grad()

        real_target = Variable(torch.FloatTensor(real_data.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
        fake_target = Variable(torch.FloatTensor(fake_data.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

        data = torch.cat((real_data, fake_data), 0)
        target = torch.cat((real_target, fake_target), 0)

        output, _, _ = self.discriminator['model'](data)
        loss = self.discriminator['loss_fn'](output, target)

        loss.backward()
        self.discriminator['optimizer'].step()

        self.discriminator['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.discriminator['writer'].add_scalar('loss', loss.item())

        metrics = self._eval_metrics(self.discriminator['metric_fns'], self.discriminator['writer'], output, target)

        return loss.item(), metrics

    def _train_encoder(self, epoch, batch_idx, batch_size):
        self.generator['model'].train()
        self.discriminator['model'].train()

        self.encoder['optimizer'].zero_grad()

        # Sample noise as generator input
        z, cat_c, cont_c = self._generate_random_input(batch_size)
        cat_target = torch.argmax(cat_c, dim=1).long()

        # Generate a batch of images
        gen_samples = self.generator['model'](z, cat_c, cont_c)
        _, cat_output, cont_output = self.discriminator['model'](gen_samples)

        cat_loss = self.encoder['cat_loss_fn'](cat_output, cat_target)
        cont_loss = self.encoder['cont_loss_fn'](cont_output, cont_c)

        loss = self.lambda_cat * cat_loss + self.lambda_con * cont_loss

        loss.backward()
        self.encoder['optimizer'].step()

        self.encoder['writer'].set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.encoder['writer'].add_scalar('loss', loss.item())

        # TODO :: might be needed
        # metrics = self._eval_metrics(self.encoder['cat_metrics_fns'], self.encoder['writer'], cat_output, cat_target)
        # metrics = self._eval_metrics(self.encoder['cont_metrics_fns'], self.encoder['writer'], cont_output, cont_c)

        return loss.item(), None

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

        total_gen_loss = 0
        total_gen_metrics = np.zeros(len(self.generator['metric_fns']))

        total_dis_loss = 0
        total_dis_metrics = np.zeros(len(self.discriminator['metric_fns']))

        total_enc_loss = 0

        for batch_idx, (data, _) in enumerate(self.data_loader):
            data = data.to(self.device)

            gen_loss, gen_metrics, gen_samples = self._train_generator(epoch, batch_idx, data.size(0))

            total_gen_loss += gen_loss
            total_gen_metrics += gen_metrics

            dis_loss, dis_metrics = self._train_discriminator(epoch, batch_idx, data, gen_samples)

            total_dis_loss += dis_loss
            total_dis_metrics += dis_metrics

            enc_loss, _ = self._train_encoder(epoch, batch_idx, data.size(0))

            total_enc_loss += enc_loss

            if batch_idx % self.log_step == 0:
                self.logger.debug(('Train Epoch: {} {} \n\t'
                    + 'generator loss\t\t: {:.6f} \n\t'
                    + 'discriminator loss\t: {:.6f} \n\t'
                    + 'encoder loss\t\t: {:.6f}').format(
                    epoch,
                    self._progress(batch_idx),
                    gen_loss,
                    dis_loss,
                    enc_loss))
                self.generator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                self.discriminator['writer'].add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = {
            'generator': {
                'loss': total_gen_loss / self.len_epoch,
            },
            'discriminator': {
                'loss': total_dis_loss / self.len_epoch,
            },
            'encoder': {
                'loss': total_enc_loss / self.len_epoch,
            }
        }

        metrics_list = (total_gen_metrics / self.len_epoch).tolist()
        metrics = {}
        for i, metric in enumerate(self.generator['metric_fns']):
            metrics[metric.__name__] = metrics_list[i]

        log['generator'].update(metrics)

        metrics_list = (total_dis_metrics / self.len_epoch).tolist()
        metrics = {}
        for i, metric in enumerate(self.discriminator['metric_fns']):
            metrics[metric.__name__] = metrics_list[i]

        log['discriminator'].update(metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log['generator'].update(val_log['generator'])
            log['discriminator'].update(val_log['discriminator'])
            log['encoder'].update(val_log['encoder'])

        if self.generator['lr_scheduler'] is not None:
            self.generator['lr_scheduler'].step()

        if self.discriminator['lr_scheduler'] is not None:
            self.discriminator['lr_scheduler'].step()

        if self.encoder['lr_scheduler'] is not None:
            self.encoder['lr_scheduler'].step()

        return log

    def _valid_epoch(self, epoch):
        val_log = self.evaluate()

        gen_val_log = {}
        for key, value in val_log['generator'].items():
            gen_val_log['val_'+key] = val_log['generator'][key]

        self.generator['writer'].set_step(epoch * len(self.valid_data_loader), 'valid')
        for key, value in gen_val_log.items():
            self.generator['writer'].add_scalar('val_'+key, value)

        val_log['generator'] = gen_val_log

        dis_val_log = {}
        for key, value in val_log['discriminator'].items():
            dis_val_log['val_'+key] = val_log['discriminator'][key]

        self.discriminator['writer'].set_step(epoch * len(self.valid_data_loader), 'valid')
        for key, value in dis_val_log.items():
            self.discriminator['writer'].add_scalar('val_'+key, value)

        val_log['discriminator'] = dis_val_log

        enc_val_log = {}
        for key, value in val_log['encoder'].items():
            enc_val_log['val_'+key] = val_log['encoder'][key]

        self.encoder['writer'].set_step(epoch * len(self.valid_data_loader), 'valid')
        for key, value in enc_val_log.items():
            self.encoder['writer'].add_scalar('val_'+key, value)

        val_log['generator'] = gen_val_log
        val_log['discriminator'] = dis_val_log
        val_log['encoder'] = enc_val_log

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
