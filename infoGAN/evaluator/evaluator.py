import numpy as np
import torch
from base import BaseEvaluator
from torch.autograd import Variable


class Evaluator(BaseEvaluator):
    """
    Evaluator class

    Note:
        Inherited from BaseEvaluator.
    """
    def __init__(self, config, logger, generator, discriminator, encoder, data_loader=None):
        super().__init__(config, logger, generator, discriminator, encoder)
        self.config = config
        self.data_loader = data_loader

    def _eval_metrics(self, metrics, writer, output, target):
        acc_metrics = np.zeros(len(metrics))
        for i, metric in enumerate(metrics):
            acc_metrics[i] += metric(output, target)
            writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _eval_generator(self, batch_idx, batch_size):

        # generate samples which can fool discriminator
        target = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(self.device)

        # Sample noise as generator input
        z, cat_c, cont_c = self._generate_random_input(batch_size)

        with torch.no_grad():
            gen_samples = self.generator['model'](z, cat_c, cont_c)
            dis_output, _, _ = self.discriminator['model'](gen_samples)
            loss = self.generator['loss_fn'](dis_output, target)

        gen_samples = gen_samples.detach()

        metrics = np.zeros(len(self.generator['metric_fns']))
        for i, metric in enumerate(self.generator['metric_fns']):
            metrics[i] += metric(dis_output, target)

        return loss.item(), metrics, gen_samples

    def _eval_discriminator(self, batch_idx, real_data, fake_data):
        real_target = Variable(torch.FloatTensor(real_data.size(0), 1).fill_(1.0), requires_grad=False).to(self.device)
        fake_target = Variable(torch.FloatTensor(fake_data.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)

        data = torch.cat((real_data, fake_data), 0)
        target = torch.cat((real_target, fake_target), 0)

        with torch.no_grad():
            output, _, _ = self.discriminator['model'](data)
            loss = self.discriminator['loss_fn'](output, target)

        metrics = np.zeros(len(self.discriminator['metric_fns']))
        for i, metric in enumerate(self.discriminator['metric_fns']):
            metrics[i] += metric(output, target)

        return loss.item(), metrics

    def _eval_encoder(self, batch_idx, batch_size):
        # Sample noise as generator input
        z, cat_c, cont_c = self._generate_random_input(batch_size)
        cat_target = torch.argmax(cat_c, dim=1).long()

        with torch.no_grad():
            gen_samples = self.generator['model'](z, cat_c, cont_c)
            _, cat_output, cont_output = self.discriminator['model'](gen_samples)

            cat_loss = self.encoder['cat_loss_fn'](cat_output, cat_target)
            cont_loss = self.encoder['cont_loss_fn'](cont_output, cont_c)

            loss = self.lambda_cat * cat_loss + self.lambda_con * cont_loss

        # TODO :: might be needed
        # metrics = self._eval_metrics(self.encoder['cat_metrics_fns'], self.encoder['writer'], cat_output, cat_target)
        # metrics = self._eval_metrics(self.encoder['cont_metrics_fns'], self.encoder['writer'], cont_output, cont_c)

        return loss.item(), None

    def evaluate(self):
        """
        Testing logic

        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """

        self.generator['model'].eval()
        self.discriminator['model'].eval()

        total_gen_loss = 0
        total_gen_metrics = np.zeros(len(self.generator['metric_fns']))

        total_dis_loss = 0
        total_dis_metrics = np.zeros(len(self.discriminator['metric_fns']))

        total_enc_loss = 0

        for batch_idx, (data, _) in enumerate(self.data_loader):
            data = data.to(self.device)

            gen_loss, gen_metrics, gen_samples = self._eval_generator(batch_idx, data.size(0))

            total_gen_loss += gen_loss
            total_gen_metrics += gen_metrics

            dis_loss, dis_metrics = self._eval_discriminator(batch_idx, data, gen_samples)

            total_dis_loss += dis_loss
            total_dis_metrics += dis_metrics

            enc_loss, _ = self._eval_encoder(batch_idx, data.size(0))

            total_enc_loss += enc_loss

        num_batch = len(self.data_loader)

        log = {
            'generator': {
                'loss': total_gen_loss / num_batch,
            },
            'discriminator': {
                'loss': total_dis_loss / num_batch,
            },
            'encoder': {
                'loss': total_enc_loss / num_batch,
            }
        }

        metrics_list = (total_gen_metrics / num_batch).tolist()
        metrics = {}
        for i, metric in enumerate(self.generator['metric_fns']):
            metrics[metric.__name__] = metrics_list[i]

        log['generator'].update(metrics)

        metrics_list = (total_dis_metrics / num_batch).tolist()
        metrics = {}
        for i, metric in enumerate(self.discriminator['metric_fns']):
            metrics[metric.__name__] = metrics_list[i]

        log['discriminator'].update(metrics)

        return log
