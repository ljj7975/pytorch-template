import argparse
import collections
import torch
import itertools
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as model_arch
from parse_config import ConfigParser
from trainer import Trainer
from torch.autograd import Variable
from utils import save_generated_images, GifGenerator


def main(config):
    '''===== Verify Config ====='''
    assert config['generator']['arch']['args']['img_shape'] == \
        config['discriminator']['arch']['args']['img_shape']
    assert config['generator']['arch']['args']['n_classes'] == \
        config['discriminator']['arch']['args']['n_classes']
    assert config['generator']['arch']['args']['code_dim'] == \
        config['discriminator']['arch']['args']['code_dim']

    logger = config.get_logger('train', config['trainer']['verbosity'])

    '''===== Data Loader ====='''
    logger.info('preparing data loader')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    '''===== Generator ====='''
    logger.info('preparing Generator')

    # build model architecture, then print to console
    generator_model = config.initialize('arch', model_arch, 'generator')
    logger.info(generator_model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['generator']['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['generator']['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator_model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, 'generator', trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'generator', optimizer)

    generator = {
        'model': generator_model,
        'loss_fn': loss_fn,
        'metric_fns': metric_fns,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }

    '''===== Discriminator ====='''
    logger.info('preparing Discriminator')

    # build model architecture, then print to console
    discriminator_model = config.initialize('arch', model_arch, 'discriminator')
    logger.info(discriminator_model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['discriminator']['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['discriminator']['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, discriminator_model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, 'discriminator', trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'discriminator', optimizer)

    discriminator = {
        'model': discriminator_model,
        'loss_fn': loss_fn,
        'metric_fns': metric_fns,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }

    '''===== Encoder ====='''
    logger.info('preparing Encoder')

    cat_loss_fn = getattr(module_loss, config['encoder']['categorical_loss'])
    cont_loss_fn = getattr(module_loss, config['encoder']['continuous_loss'])

    cat_metric_fns = [getattr(module_metric, met) for met in config['encoder']['cat_metrics']]
    cont_metric_fns = [getattr(module_metric, met) for met in config['encoder']['cont_metrics']]

    params = itertools.chain(generator_model.parameters(), discriminator_model.parameters())
    trainable_params = filter(lambda p: p.requires_grad, params)

    optimizer = config.initialize('optimizer', torch.optim, 'encoder', trainable_params)
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'encoder', optimizer)

    encoder = {
        'cat_loss_fn': cat_loss_fn,
        'cont_loss_fn': cont_loss_fn,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'cat_metric_fns': cat_metric_fns,
        'cont_metric_fns': cont_metric_fns,
    }


    '''===== Training ====='''

    logger.info("< TRAINING >")

    gif_generator = GifGenerator(str(config.output_dir / "progress.gif"))

    trainer = Trainer(config, logger, generator, discriminator, encoder, gif_generator, data_loader, valid_data_loader)

    trainer.train()

    '''===== Testing ====='''

    logger.info("< TESTING >")

    log = trainer.evaluate()

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

    log_msg += '    Encoder :\n'
    for key, value in log['encoder'].items():
        if isinstance(value, float):
            value = round(value, 6)
        log_msg += '        {:15s}: {}'.format(str(key), value) + '\n'

    logger.info(log_msg)

    '''===== Generate samples ====='''

    logger.info("< SAMPLE GENERATION >")

    trainer.generate_image()
    gif_generator.save()

    logger.info("saved progress as a gif at {}".format(gif_generator.gif_name))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='infoGAN PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', nargs=2, default=None, type=str,
                      help='paths to generator, discriminator and encoder checkpoint (default: None)')
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
