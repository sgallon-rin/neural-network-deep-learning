#!/bin/bash
conda activate sjl
cd /home/jupyter/sjl/neural-network-deep-learning/project2/code
# try network structure conv1
#python train_resnet.py --conv1 723
#python train_resnet.py --conv1 721
#python train_resnet.py --conv1 711
#python train_resnet.py --conv1 521
#python train_resnet.py --conv1 511
## below is baseline
#python train_resnet.py --conv1 311

# try aug
## default python train_resnet.py --aug 0 --conv1 311
#python train_resnet.py --aug 1 --conv1 311
#python train_resnet.py --aug 2 --conv1 311

# try activation
## default python train_resnet.py --activation relu --conv1 311
#python train_resnet.py --activation elu --conv1 311
#python train_resnet.py --activation leakyrelu --conv1 311
#python train_resnet.py --activation rrelu --conv1 311
python train_resnet.py --activation sigmoid --conv1 311
python train_resnet.py --activation tanh --conv1 311

# try optimizer
## default python train_resnet.py --optim sgd --conv1 311
python train_resnet.py --optim adagrad --conv1 311
python train_resnet.py --optim rmsprop --conv1 311
python train_resnet.py --optim adadelta --conv1 311
python train_resnet.py --optim adam --conv1 311

# try loss func L2 reg
python train_resnet.py --optim adam --reg 0.001

# try hidden
python train_resnet.py --conv1 311 --hidden 10
python train_resnet.py --conv1 311 --hidden 20
python train_resnet.py --conv1 311 --hidden 50
python train_resnet.py --conv1 311 --hidden 100

# try scheduler
