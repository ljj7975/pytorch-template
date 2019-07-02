import os
import torch
from abc import abstractmethod
import numpy as np
from numpy import inf
from logger import TensorboardWriter
from tqdm import tqdm
from torch.autograd import Variable
from utils import SampleGenerator, to_categorical, init_weights_to_normal
from evaluator import Evaluator

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, logger, generator, discriminator, encoder, valid_data_loader=None):
        self.config = config
        self.logger = logger
        self.evaluator = Evaluator(config, logger, generator, discriminator, encoder, valid_data_loader)

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

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.early_stop = cfg_trainer.get('early_stop', inf)
        self.start_epoch = 1

        self.latent_dim = config['generator']['arch']['args']['latent_dim']
        self.cat_dim = config['generator']['arch']['args']['cat_dim']
        self.cont_dim = config['generator']['arch']['args']['cont_dim']

        self.sample_generator = SampleGenerator(config.output_dir, generator['model'], self.latent_dim, self.cat_dim, self.cont_dim, n_row=self.cat_dim)

        self.generator['model'].apply(init_weights_to_normal)
        self.discriminator['model'].apply(init_weights_to_normal)

        self.generator['monitor'] = self.initialize_monitor(cfg_trainer['monitor'].get('generator', 'off'))
        self.discriminator['monitor'] = self.initialize_monitor(cfg_trainer['monitor'].get('discriminator', 'off'))
        self.encoder['monitor'] = self.initialize_monitor(cfg_trainer['monitor'].get('encoder', 'off'))

        self.generator['checkpoint_dir'] = config.save_dir / "generator"
        self.generator['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
        self.discriminator['checkpoint_dir'] = config.save_dir / "discriminator"
        self.discriminator['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
        self.encoder['checkpoint_dir'] = config.save_dir / "encoder"
        self.encoder['checkpoint_dir'].mkdir(parents=True, exist_ok=True)

        self.generator['writer'] = TensorboardWriter(config.log_dir / 'generator', self.logger, cfg_trainer['tensorboard'])
        self.discriminator['writer'] = TensorboardWriter(config.log_dir / 'discriminator', self.logger, cfg_trainer['tensorboard'])
        self.encoder['writer'] = TensorboardWriter(config.log_dir / 'encoder', self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint('generator', config.resume['generator'])
            self._resume_checkpoint('discriminator', config.resume['discriminator'])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
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

    def initialize_training(self, player):
        player['model'] = player['model'].to(self.device)
        if len(self.device_ids) > 1:
            player['model'] = torch.nn.DataParallel(player['model'], device_ids=device_ids)

        return player

    def initialize_monitor(self, monitor):
        if monitor == 'off':
            return None

        mnt_mode, mnt_metric = monitor.split()
        assert mnt_mode in ['min', 'max']

        return {
            "mnt_mode": mnt_mode,
            "mnt_metric": mnt_metric,
            "mnt_best": inf if mnt_mode == 'min' else -inf,
            "not_improved_count": 0
        }

    def train(self):
        """
        Full training logic
        """
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            log_msg = '< Epoch {} >'.format(log['epoch']) + '\n'
            log_msg += '    Generator :\n'
            for key, value in log['generator'].items():
                if isinstance(value, float):
                    value = round(value, 6)
                log_msg += '        {:15s}: {}'.format(str(key), value) + '\n'

            log_msg += '    discriminator :\n'
            for key, value in log['discriminator'].items():
                if isinstance(value, float):
                    value = round(value, 6)
                log_msg += '        {:15s}: {}'.format(str(key), value) + '\n'

            log_msg += '    encoder :\n'
            for key, value in log['encoder'].items():
                if isinstance(value, float):
                    value = round(value, 6)
                log_msg += '        {:15s}: {}'.format(str(key), value) + '\n'

            self.logger.info(log_msg)

            gen_early_stop = self._check_and_save(epoch, 'generator', log['generator'])
            dis_early_stop = self._check_and_save(epoch, 'discriminator', log['discriminator'])
            _, enc_early_stop = self._monitor_progress('encoder', log['encoder'])

            if gen_early_stop or dis_early_stop or enc_early_stop:
                break;

        self.sample_generator.generate_gif()

    def _check_and_save(self, epoch, player, log):
        # evaluate model performance according to configured metric, save best checkpoint as model_best
        best, early_stop = self._monitor_progress(player, log)

        if early_stop:
            self.logger.info("Validation performance of {} didn\'t improve for {} epochs. "
                             "Training stops.".format(player, self.early_stop))

        if epoch % self.save_period == 0 or best:
            self._save_checkpoint(epoch, player, save_best=best)
            if player == "generator":
                self.sample_generator.generate_sample(str(epoch))

        return early_stop

    def _monitor_progress(self, player, log):
        if player == 'generator':
            monitor = self.generator['monitor']
        elif player == 'discriminator':
            monitor = self.discriminator['monitor']
        elif player == 'encoder':
            monitor = self.encoder['monitor']

        best = False
        if monitor['mnt_mode'] != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (monitor['mnt_mode'] == 'min' and log[monitor['mnt_metric']] <= monitor['mnt_best']) or \
                           (monitor['mnt_mode'] == 'max' and log[monitor['mnt_metric']] >= monitor['mnt_best'])
            except KeyError:
                self.logger.warning("Warning: Metric '{}' is not found for {}. "
                                    "Model performance monitoring is disabled.".format(monitor['mnt_metric'], player))
                monitor['mnt_mode'] = 'off'
                improved = False
                monitor['not_improved_count'] = 0

            if improved:
                monitor['mnt_best'] = log[monitor['mnt_metric']]
                monitor['not_improved_count'] = 0
                best = True
            else:
                monitor['not_improved_count'] += 1

        return best, monitor['not_improved_count'] > self.early_stop

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

    def _save_checkpoint(self, epoch, player, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if player == 'generator':
            player_info = self.generator
        elif player == 'discriminator':
            player_info = self.discriminator

        arch = type(player_info['model']).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': player_info['model'].state_dict(),
            'optimizer': player_info['optimizer'].state_dict(),
            'monitor_best': player_info['monitor']['mnt_best'],
            'config': player_info['config']
        }
        filename = str(player_info['checkpoint_dir'] / 'checkpoint-epoch{}.pth').format(epoch)
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(player_info['checkpoint_dir'] / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best {}: model_best.pth ...".format(player))

    def _resume_checkpoint(self, player, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint for {} : {} ...".format(player, resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = max(checkpoint['epoch'] + 1, self.start_epoch)
        self.generator['monitor']['mnt_best'] = checkpoint['monitor_best']

        if player == 'generator':
            model = self.generator['model']
            optimizer = self.generator['optimizer']
        elif player == 'discriminator':
            model = self.discriminator['model']
            optimizer = self.discriminator['optimizer']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config[player]['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config[player]['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded for {}. Resume training from epoch {}".format(player, self.start_epoch))

    def evaluate(self):
        return self.evaluator.evaluate()
