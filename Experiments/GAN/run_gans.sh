#!/bin/bash
python train.py configs/fr_default.yaml 0.0001 1 0.99 cifar10 output/cifar_sim1_run1 150000 10 1
python train.py configs/fr_default.yaml 0.0001 2 0.99 cifar10 output/cifar_sim2_run1 150000 10 1
python train.py configs/fr_default.yaml 0.0001 4 0.99 cifar10 output/cifar_sim3_run1 150000 10 1
python train.py configs/fr_default.yaml 0.0001 8 0.99 cifar10 output/cifar_sim4_run1 150000 10 1

python train.py configs/fr_default.yaml 0.0001 1 0.99 image output/celeba_sim1_run1 150000 10 1
python train.py configs/fr_default.yaml 0.0001 2 0.99 image output/celeba_sim2_run1 150000 10 1
python train.py configs/fr_default.yaml 0.0001 4 0.99 image output/celeba_sim3_run1 150000 10 1
python train.py configs/fr_default.yaml 0.0001 8 0.99 image output/celeba_sim4_run1 150000 10 1

python train.py configs/fr_default.yaml 0.0001 1 0.99 cifar10 output/cifar_sim1_run2 150000 10 2 
python train.py configs/fr_default.yaml 0.0001 2 0.99 cifar10 output/cifar_sim2_run2 150000 10 2
python train.py configs/fr_default.yaml 0.0001 4 0.99 cifar10 output/cifar_sim3_run2 150000 10 2
python train.py configs/fr_default.yaml 0.0001 8 0.99 cifar10 output/cifar_sim4_run2 150000 10 2

python train.py configs/fr_default.yaml 0.0001 1 0.99 image output/celeba_sim1_run2 150000 10 2
python train.py configs/fr_default.yaml 0.0001 2 0.99 image output/celeba_sim2_run2 150000 10 2
python train.py configs/fr_default.yaml 0.0001 4 0.99 image output/celeba_sim3_run2 150000 10 2
python train.py configs/fr_default.yaml 0.0001 8 0.99 image output/celeba_sim4_run2 150000 10 2

python train.py configs/fr_default.yaml 0.0001 1 0.99 cifar10 output/cifar_sim1_run3 150000 10 3 
python train.py configs/fr_default.yaml 0.0001 2 0.99 cifar10 output/cifar_sim2_run3 150000 10 3
python train.py configs/fr_default.yaml 0.0001 4 0.99 cifar10 output/cifar_sim3_run3 150000 10 3
python train.py configs/fr_default.yaml 0.0001 8 0.99 cifar10 output/cifar_sim4_run3 150000 10 3

python train.py configs/fr_default.yaml 0.0001 1 0.99 image output/celeba_sim1_run3 150000 10 3
python train.py configs/fr_default.yaml 0.0001 2 0.99 image output/celeba_sim2_run3 150000 10 3
python train.py configs/fr_default.yaml 0.0001 4 0.99 image output/celeba_sim3_run3 150000 10 3
python train.py configs/fr_default.yaml 0.0001 8 0.99 image output/celeba_sim4_run3 150000 10 3

python train.py configs/fr_default.yaml 0.0001 1 0.99 cifar10 output/cifar_sim1_run1_reg1 300000 1 1
python train.py configs/fr_default.yaml 0.0001 2 0.99 cifar10 output/cifar_sim2_run1_reg1 300000 1 1
python train.py configs/fr_default.yaml 0.0001 4 0.99 cifar10 output/cifar_sim3_run1_reg1 300000 1 1
python train.py configs/fr_default.yaml 0.0001 8 0.99 cifar10 output/cifar_sim4_run1_reg1 300000 1 1

python train.py configs/fr_default.yaml 0.0001 1 0.99 image output/celeba_sim1_run1_reg1 300000 1 1
python train.py configs/fr_default.yaml 0.0001 2 0.99 image output/celeba_sim2_run1_reg1 300000 1 1
python train.py configs/fr_default.yaml 0.0001 4 0.99 image output/celeba_sim3_run1_reg1 300000 1 1
python train.py configs/fr_default.yaml 0.0001 8 0.99 image output/celeba_sim4_run1_reg1 300000 1 1

python train.py configs/fr_default.yaml 0.0001 1 0.99 cifar10 output/cifar_sim1_run2_reg1 300000 1 2
python train.py configs/fr_default.yaml 0.0001 2 0.99 cifar10 output/cifar_sim2_run2_reg1 300000 1 2
python train.py configs/fr_default.yaml 0.0001 4 0.99 cifar10 output/cifar_sim3_run2_reg1 300000 1 2
python train.py configs/fr_default.yaml 0.0001 8 0.99 cifar10 output/cifar_sim4_run2_reg1 300000 1 2

python train.py configs/fr_default.yaml 0.0001 1 0.99 image output/celeba_sim1_run2_reg1 300000 1 2
python train.py configs/fr_default.yaml 0.0001 2 0.99 image output/celeba_sim2_run2_reg1 300000 1 2
python train.py configs/fr_default.yaml 0.0001 4 0.99 image output/celeba_sim3_run2_reg1 300000 1 2
python train.py configs/fr_default.yaml 0.0001 8 0.99 image output/celeba_sim4_run2_reg1 300000 1 2

python train.py configs/fr_default.yaml 0.0001 1 0.99 cifar10 output/cifar_sim1_run3_reg1 300000 1 3
python train.py configs/fr_default.yaml 0.0001 2 0.99 cifar10 output/cifar_sim2_run3_reg1 300000 1 3
python train.py configs/fr_default.yaml 0.0001 4 0.99 cifar10 output/cifar_sim3_run3_reg1 300000 1 3
python train.py configs/fr_default.yaml 0.0001 8 0.99 cifar10 output/cifar_sim4_run3_reg1 300000 1 3

python train.py configs/fr_default.yaml 0.0001 1 0.99 image output/celeba_sim1_run3_reg1 300000 1 3
python train.py configs/fr_default.yaml 0.0001 2 0.99 image output/celeba_sim2_run3_reg1 300000 1 3
python train.py configs/fr_default.yaml 0.0001 4 0.99 image output/celeba_sim3_run3_reg1 300000 1 3
python train.py configs/fr_default.yaml 0.0001 8 0.99 image output/celeba_sim4_run3_reg1 300000 1 3


