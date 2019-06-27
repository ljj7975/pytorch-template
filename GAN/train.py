import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as model_arch
from parse_config import ConfigParser
from trainer import Trainer, evaluate


def main(config):
    logger = config.get_logger('GAN', config['trainer']['verbosity'])

    '''===== Data Loader ====='''
    logger.info('preparing data loader')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    '''===== Generator ====='''
    logger.info('preparing Generator')
    # gen_logger = config.get_logger('generator')

    # build model architecture, then print to console
    model = config.initialize('arch', model_arch, 'generator')
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['generator']['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['generator']['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, 'generator', trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'generator', optimizer)

    generator = {
        'model': model,
        'loss_fn': loss_fn,
        'metric_fns': metric_fns,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }

    '''===== Discriminator ====='''
    logger.info('preparing Discriminator')
    # dis_logger = config.get_logger('discriminator')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', model_arch, 'discriminator')
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['discriminator']['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['discriminator']['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, 'discriminator', trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'discriminator', optimizer)

    discriminator = {
        # 'logger': dis_logger,
        'model': model,
        'loss_fn': loss_fn,
        'metric_fns': metric_fns,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }

    '''===== Training ====='''

    trainer = Trainer(config, logger, generator, discriminator, data_loader, valid_data_loader)

    trainer.train()

    '''===== Testing ====='''

    log = evaluate(generator, discriminator, config, valid_data_loader)

    log_msg = '< Evaluation >\n'
    log_msg += '    Generator :\n'
    for key, value in log['generator'].items():
        if isinstance(value, float):
            value = round(value, 6)
        log_msg += '        {:15s}: {}'.format(str(key), value) + '\n'

    log_msg += '    Discriminator :\n'
    for key, value in log['discriminator'].items():
        if isinstance(value, float):
            value = round(value, 6)
        log_msg += '        {:15s}: {}'.format(str(key), value) + '\n'

    logger.info(log_msg)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', nargs=2, default=None, type=str,
                      help='paths to generator and discriminator checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
