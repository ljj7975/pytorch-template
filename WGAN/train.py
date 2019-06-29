import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as model_arch
from parse_config import ConfigParser
from trainer import Trainer, evaluate
from torch.autograd import Variable
from utils import save_generated_images, GifGenerator


def main(config):
    logger = config.get_logger('train', config['trainer']['verbosity'])

    '''===== Data Loader ====='''
    logger.info('preparing data loader')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    '''===== Generator ====='''
    logger.info('preparing Generator')

    # build model architecture, then print to console
    model = config.initialize('arch', model_arch, 'generator')
    logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, 'generator', trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'generator', optimizer)

    generator = {
        'model': model,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }

    '''===== Discriminator ====='''
    logger.info('preparing Discriminator')

    # build model architecture, then print to console
    model = config.initialize('arch', model_arch, 'discriminator')
    logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, 'discriminator', trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, 'discriminator', optimizer)

    discriminator = {
        'model': model,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler
    }

    gif_generator = GifGenerator(str(config.output_dir / "progress.gif"))

    '''===== Training ====='''

    trainer = Trainer(config, logger, generator, discriminator, gif_generator, data_loader, valid_data_loader)

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

    '''===== Generate samples ====='''

    num_samples = 16
    if trainer.default_z:
        z = trainer.default_z
    else:
        if num_samples != trainer.num_samples and trainer.num_samples > 0:
            num_samples = trainer.num_samples
        z = Variable(torch.randn(num_samples, trainer.z_size))

    z = z.to(trainer.device)

    generator['model'].eval()
    samples = generator['model'](z).detach().cpu()
    output_file = str(config.output_dir / "final.png")

    img = save_generated_images(samples.detach(), output_file)
    logger.info("saved generated images at {}".format(output_file))

    gif_generator.add_image(img)
    gif_generator.save()

    logger.info("saved progress as a gif at {}".format(gif_generator.gif_name))

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
