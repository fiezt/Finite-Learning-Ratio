import torchvision
import torch 
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from gan_training.inputs import get_dataset

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists('cifar_real'):
    os.makedirs('cifar_real')

    train_dataset, nlabels = get_dataset(
        name='cifar10',
        data_dir='../data',
        size=32
    )

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=10000,
            num_workers=16,
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )

    for x_real, y in train_loader:
        break

    x_real = x_real/2 + 0.5

    for i in range(x_real.size(0)):
        torchvision.utils.save_image(x_real[i, :, :, :], 'cifar_real/{}.png'.format(i))
        
        
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists('celeb_real'):
    os.makedirs('celeb_real')

    train_dataset, nlabels = get_dataset(
        name='image',
        data_dir='../data',
        size=32
    )

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=10000,
            num_workers=16,
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )

    for x_real, y in train_loader:
        break

    x_real = x_real/2 + 0.5

    for i in range(x_real.size(0)):
        torchvision.utils.save_image(x_real[i, :, :, :], 'celeb_real/{}.png'.format(i))

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists('z_data.npy'):
    zdist = get_zdist('gauss', 256, device='cpu')
    ztest = zdist.sample((50000,))
    np.save('z_data.npy', ztest.cpu().numpy())