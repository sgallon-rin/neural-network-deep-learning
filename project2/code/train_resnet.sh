#!/bin/bash
conda activate sjl
cd /home/jupyter/sjl/neural-network-deep-learning/project2/code
# try network structure conv1
python train_resnet.py --conv1 723
python train_resnet.py --conv1 721
python train_resnet.py --conv1 711
python train_resnet.py --conv1 521
python train_resnet.py --conv1 511
# below is baseline
python train_resnet.py --conv1 311

# try aug
python train_resnet.py --aug 1 --conv1 311
python train_resnet.py --aug 2 --conv1 311

# try activation
python train_resnet.py --activation elu --conv1 311
python train_resnet.py --activation leakyrelu --conv1 311
python train_resnet.py --activation rrelu --conv1 311
python train_resnet.py --activation sigmoid --conv1 311
python train_resnet.py --activation tanh --conv1 311

# try optimizer
python train_resnet.py --optimizer adagrad --conv1 311
python train_resnet.py --optimizer rmsprop --conv1 311
python train_resnet.py --optimizer adadelta --conv1 311
python train_resnet.py --optimizer adam --conv1 311

# try hidden

# try scheduler
