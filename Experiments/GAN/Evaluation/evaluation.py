import os
import copy
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torchvision
from fid_score import calculate_fid_given_paths
import sys
import pickle
sys.path.append(os.path.join(os.getcwd(), '..'))
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models)

def perform_evaluation(run_name, image_type):
    
    out_dir = os.path.join(os.getcwd(), '..', 'output', run_name)
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*')))
    evaluation_dict = {}

    for point in checkpoints:
        if not int(point.split('/')[-1].split('_')[1].split('.')[0]) % 10000 == 0:
            continue

        iter_num = int(point.split('/')[-1].split('_')[1].split('.')[0])
        model_file = point.split('/')[-1]

        config = load_config('../configs/fr_default.yaml', None)
        is_cuda = (torch.cuda.is_available())
        checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
        device = torch.device("cuda:0" if is_cuda else "cpu")

        generator, discriminator = build_models(config)

        # Put models on gpu if needed
        generator = generator.to(device)
        discriminator = discriminator.to(device)

        # Use multiple GPUs if possible
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

        generator_test_9 = copy.deepcopy(generator)
        generator_test_99 = copy.deepcopy(generator)
        generator_test_999 = copy.deepcopy(generator)
        generator_test_9999 = copy.deepcopy(generator)

        # Register modules to checkpoint
        checkpoint_io.register_modules(
            generator=generator,
            generator_test_9=generator_test_9,
            generator_test_99=generator_test_99,
            generator_test_999=generator_test_999,
            generator_test_9999=generator_test_9999,
            discriminator=discriminator,)

        # Load checkpoint 
        load_dict = checkpoint_io.load(model_file)
        
        # Distributions
        ydist = get_ydist(config['data']['nlabels'], device=device)
        zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
        z_sample =  torch.Tensor(np.load('z_data.npy')).to(device)

        #for name, model in zip(['0_', '09_', '099_', '0999_', '09999_'], [generator, generator_test_9, generator_test_99, generator_test_999, generator_test_9999]):
        for name, model in zip(['099_', '0999_', '09999_'], [generator_test_99, generator_test_999, generator_test_9999]):

            # Evaluator
            evaluator = Evaluator(model, zdist, ydist, device=device)

            x_sample = []

            for i in range(10):
                x = evaluator.create_samples(z_sample[i*1000:(i+1)*1000])
                x_sample.append(x)

            x_sample = torch.cat(x_sample)
            x_sample = x_sample/2 + 0.5

            if not os.path.exists('fake_data'):
                os.makedirs('fake_data')
                
            for i in range(10000):
                torchvision.utils.save_image(x_sample[i, :, :, :], 'fake_data/{}.png'.format(i))

            fid_score = calculate_fid_given_paths(['fake_data', image_type+'_real'], 50, True, 2048)
            print(iter_num, name, fid_score)

            os.system("rm -rf " + "fake_data")

            evaluation_dict[(iter_num, name[:-1])] = {'FID':fid_score}
            
            if not os.path.exists('evaluation_data/'+run_name):
                os.makedirs('evaluation_data/'+run_name)

            pickle.dump(evaluation_dict, open('evaluation_data/'+run_name+'/eval_fid.p', 'wb'))
            
            
if __name__ == "__main__": 

    image_type = 'cifar'
    for run_name in ['cifar_sim1_run1_reg1', 'cifar_sim2_run1_reg1', 'cifar_sim3_run1_reg1', 'cifar_sim4_run1_reg1']:
        perform_evaluation(run_name, image_type)        
        
    image_type = 'celeb'
    for run_name in ['celeba_sim1_run1_reg1', 'celeba_sim2_run1_reg1', 'celeba_sim3_run1_reg1', 'celeba_sim4_run1_reg1']:
        perform_evaluation(run_name, image_type)
    
    
    
    
    
