import argparse
import os
from os import path
import time
import glob
import copy
import torch
from torch import nn
import numpy as np
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler,
)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('lr', type=float, help='learning rate')
parser.add_argument('tau', type=float, help='timescale separation')
parser.add_argument('alpha', type=float, help='RMSProp parameter')
parser.add_argument('type', type=str, help='dataset type')
parser.add_argument('out_dir', type=str, help='output directory')
parser.add_argument('num_iter', type=int, help='number of iterations')
parser.add_argument('reg_param', type=float, help='regularization parameter')
parser.add_argument('random_seed', type=int, help='random seed')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
args = parser.parse_args()

config = load_config(args.config, None)

seed = args.random_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

is_cuda = (torch.cuda.is_available() and not args.no_cuda)

config['training']['lr_g'] = args.lr
config['training']['lr_d'] = args.lr*args.tau
config['training']['alpha'] = args.alpha
config['data']['type'] = args.type
config['training']['out_dir'] = args.out_dir
config['training']['reg_param'] = args.reg_param

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']

out_dir = config['training']['out_dir']
checkpoint_dir = path.join(out_dir, 'chkpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Logger
checkpoint_io = CheckpointIO(
    checkpoint_dir=checkpoint_dir
)

device = torch.device("cuda:0" if is_cuda else "cpu")


# Dataset
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size']
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
)

# Number of labels
nlabels = min(nlabels, config['data']['nlabels'])

# Create models
generator, discriminator = build_models(config)


# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

print(generator)
print(discriminator)

g_optimizer, d_optimizer = build_optimizers(
    generator, discriminator, config
)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)


# Logger
logger = Logger(
    log_dir=path.join(out_dir, 'logs'),
    img_dir=path.join(out_dir, 'imgs'),
    monitoring=config['training']['monitoring'],
    monitoring_dir=path.join(out_dir, 'monitoring')
)

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                  device=device)

# Save for tests
ntest = batch_size
x_real, ytest = utils.get_nsamples(train_loader, ntest)
ytest.clamp_(None, nlabels-1)
ztest = zdist.sample((ntest,))
utils.save_images(x_real, path.join(out_dir, 'real.png'))

# Test generator
generator_test_9 = copy.deepcopy(generator)
generator_test_99 = copy.deepcopy(generator)
generator_test_999 = copy.deepcopy(generator)
generator_test_9999 = copy.deepcopy(generator)

# Evaluator
evaluator = Evaluator(generator, zdist, ydist,
                      batch_size=batch_size, device=device)

evaluator_9 = Evaluator(generator_test_9, zdist, ydist,
                        batch_size=batch_size, device=device)

evaluator_99 = Evaluator(generator_test_99, zdist, ydist,
                         batch_size=batch_size, device=device)

evaluator_999 = Evaluator(generator_test_999, zdist, ydist,
                          batch_size=batch_size, device=device)

evaluator_9999 = Evaluator(generator_test_9999, zdist, ydist,
                           batch_size=batch_size, device=device)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    generator_test_9=generator_test_9,
    generator_test_99=generator_test_99,
    generator_test_999=generator_test_999,
    generator_test_9999=generator_test_9999,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
)

# Train
tstart = t0 = time.time()

# Load checkpoint if it exists
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*')), reverse=True)
if len(checkpoints) > 0:
    model_file = checkpoints[0].split('/')[-1]
    stats_file = 'stats_' + model_file.split('_')[1].split('.')[0] + '.p'
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    logger.load_stats(stats_file)
else:
    it = epoch_idx = -1

# Learning rate decay
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(
    generator, discriminator, g_optimizer, d_optimizer,
    gan_type=config['training']['gan_type'],
    reg_type=config['training']['reg_type'],
    reg_param=config['training']['reg_param']
)

# Training loop
print('Start training...')
start = time.time()
while it <= args.num_iter:
    epoch_idx += 1
    print('Start epoch %d...' % epoch_idx)

    for x_real, y in train_loader:
        it += 1

        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels-1)

        # Discriminator updates
        z = zdist.sample((batch_size,))
        dloss, reg = trainer.discriminator_trainstep(x_real, y, z)
        logger.add('losses', 'discriminator', dloss, it=it)
        logger.add('losses', 'regularizer', reg, it=it)

        # Generators updates
        if it % d_steps == 0:
            z = zdist.sample((batch_size,))
            gloss = trainer.generator_trainstep(y, z)
            logger.add('losses', 'generator', gloss, it=it)

            # Update average models
            update_average(generator_test_9, generator,
                           beta=0.9)
            update_average(generator_test_99, generator,
                           beta=0.99)
            update_average(generator_test_999, generator,
                           beta=0.999)
            update_average(generator_test_9999, generator,
                           beta=0.9999)
        
        # Update learning rates.
        g_scheduler.step()
        d_scheduler.step()

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'discriminator', d_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        # Print stats
        g_loss_last = logger.get_last('losses', 'generator')
        d_loss_last = logger.get_last('losses', 'discriminator')
        d_reg_last = logger.get_last('losses', 'regularizer')
        if it % 100 == 0:
            end = time.time()
            print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f, time = %.2f, lr_g=%.6f, lr_d=%.6f'
                  % (epoch_idx, it, g_loss_last, d_loss_last, d_reg_last, end-start, g_lr, d_lr))
            start = time.time()

        # (i) Sample if necessary
        if (it % config['training']['sample_every']) == 0 and it > 0:
            print('Creating samples...')
            for key, model in zip(['0', '9', '99', '999', '9999'], [evaluator, evaluator_9, evaluator_99, evaluator_999, evaluator_9999]):
                x = model.create_samples(ztest, ytest)
                logger.add_imgs(x, 'all', it, key)

        # (iii) Backup if necessary
        if it % backup_every == 0 and it > 0:
            print('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % it, it=it)
            logger.save_stats('stats_%08d.p' % it)
