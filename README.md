# Finite-Learning-Ratio

This repostiory contains the code for the experiments in the paper "Gradient Descent-Ascent Provably Converges to Strict Local
Minmax Equilibria with a Finite Timescale Separation" by Tanner Fiez and Lillian Ratliff. 

The Experiments directory contains the code. The notebooks directory contains Jupyter Notebooks that run the majority of the experiments in the paper. The utils folder contains a file with a function that computes the quantity tau* from the paper given a Jacobian matrix. 

The GAN experiments in the paper run the code from https://github.com/LMescheder/GAN_stability where we just make minor changes by changing the learning rates and the configuration files. The evaluation is done using the FID score code from https://github.com/mseitzer/pytorch-fid. I will upload the files that were changed and the configuration files shortly. 
