# Finite-Learning-Ratio

This repostiory contains the code for the experiments in the paper "Gradient Descent-Ascent Provably Converges to Strict Local
Minmax Equilibria with a Finite Timescale Separation" by Tanner Fiez and Lillian Ratliff. 

The Experiments directory contains the code. The notebooks directory contains Jupyter Notebooks that run the majority of the experiments in the paper. The utils folder contains a file with a function that computes the quantity tau* from the paper given a Jacobian matrix. 


The GAN experiments in the paper run the code from https://github.com/LMescheder/GAN_stability where we just make minor changes by changing the learning rates and the configuration files. The evaluation is done using the FID score code from https://github.com/mseitzer/pytorch-fid. We thanks those authors for the implementations that helped our work. 

We essentially only changed the https://github.com/LMescheder/GAN_stability by changing the learning rates and keeping multiple exponential averages at once. For ease of reproducibility though, we have included our version of the cloned repo.

The finiteratio.yaml provides the conda environment used for this work. 

For the GAN experiments, the data would need to be downloaded for CIFAR-10 and CelebA and put in the data folder inside the GAN folder. Then, the run_gans.sh should run the experiments we ran. Note that the experiments do take probably a couple weeks to run in full for all of the runs. The results will be stored in the output folder. The code for evaluation is in the Evaluation folder inside the GAN folder. We provide the latent data used to generate images for the FID score. The generate_evaluation_data.py creates the real data for the FID score. Then, you need to run input the names of the folders from the output folder with the saved checkpoints into the file evaluation.py. This file can then be ran to generate the FID scores along the learning path. The file evaluation.ipynb then made the plots of the FID scores in the paper.
