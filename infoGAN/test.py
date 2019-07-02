import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as model_arch
from parse_config import ConfigParser
from trainer import evaluate


def main(config):
    logger = config.get_logger('test')

    '''===== Data Loader ====='''
    logger.info('preparing data loader')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    '''===== Generator ====='''
    logger.info('preparing Generator')

    # build model architecture, then print to console
    model = config.initialize('arch', model_arch, 'generator')

    logger.info('Loading checkpoint for Generator: {} ...'.format(config.resume['generator']))
    checkpoint = torch.load(config.resume['generator'])
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['generator']['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['generator']['metrics']]

    generator = {
        'model': model,
        'loss_fn': loss_fn,
        'metric_fns': metric_fns
    }

    '''===== Discriminator ====='''
    logger.info('preparing Discriminator')

    # build model architecture, then print to console
    model = config.initialize('arch', model_arch, 'discriminator')

    logger.info('Loading checkpoint for Discriminator: {} ...'.format(config.resume['discriminator']))
    checkpoint = torch.load(config.resume['discriminator'])
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['discriminator']['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['discriminator']['metrics']]

    discriminator = {
        'model': model,
        'loss_fn': loss_fn,
        'metric_fns': metric_fns
    }

    '''===== Testing ====='''

    log = evaluate(generator, discriminator, config, data_loader)

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

    args.add_argument('-r', '--resume', nargs=2, default=None, type=str,
                      help='paths to generator and discriminator checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)