#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2021/5/27 10:55
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : train_resnet.py
# @Description : train resnet

import sys
import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
# from tqdm import tqdm
from config import DATA_ROOT, HOME
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x8d, resnext50_32x4d, \
    wide_resnet50_2, wide_resnet101_2
from logger_utils import logger

parser = argparse.ArgumentParser(description='Train a Network on CIFAR-10')
parser.add_argument('--outf', default=os.path.join(HOME, "saved_curves"),
                    help='folder to output curves')  # 输出结果保存路径，用于画图
parser.add_argument('--save-board', type=bool, default=False,
                    help='whether to save tensorboard logs')
parser.add_argument('--boardf', default=os.path.join(HOME, "saved_boards"),
                    help='folder to output tensorboard logs')  # 输出tensorboard信息保存路径
parser.add_argument('--outfname', default='',
                    help='filename of output curves')  # 输出结果保存文件名
parser.add_argument('--model-path', default=os.path.join(HOME, "saved_models"),
                    help='path to folder to save trained models')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--epoch', type=int, default=50,
                    help='maximum epoch')
parser.add_argument('--batch', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--reg', type=float, default=5e-4,
                    help='weight decay coefficient (L2 regularization)')
parser.add_argument('--aug', type=int, default=0,
                    help='data augmentation type (0/1/2)=(no aug(default)/RandomGrayscal/RandomCrop+HorizontalFlip)')
parser.add_argument('--activation', default='relu',
                    help='activation function (default relu)')
parser.add_argument('--hidden', type=int, default=0,
                    help='hidden layer size (default 0 for no hidden layer)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout probability for hidden layer (0 for no dropout, default 0.2)')
parser.add_argument('--model-name', default='resnet34',
                    help='name of the model (default resnet34)')
parser.add_argument('--optim', default='sgd',
                    help='choose which optimizer to use (default sgd)')
parser.add_argument('--scheduler', default='multisteplr',
                    help='the scheme of scheduler (default multisteplr)')  # 学习率调整策略
args = parser.parse_args()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。


set_seed(args.seed)
logger.info("Set seed: {}".format(args.seed))
# 0 HYPER PARAMETERS
MAX_EPOCH = args.epoch
logger.info("Max epoch: {}".format(MAX_EPOCH))
BATCH_SIZE = args.batch
logger.info("Batch size: {}".format(BATCH_SIZE))
LR = args.lr
logger.info("Initial learning rate: {}".format(LR))
L2_REG = args.reg
logger.info("L2 regularization weight decay coefficient: {}".format(L2_REG))
log_interval = 40  # 多少个batch打印一次学习信息
val_interval = 1  # 多少个epoch进行一次验证
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('Use device: {}, device name: {}'.format(device, torch.cuda.get_device_name(device)))

# 1 DATA
# data augmentation
if args.aug not in [0, 1, 2]:
    args.aug = 0
    logger.warning("Illegal aug: {}, aug should be 0/1/2, reset to default 0".format(args.aug))
if args.aug == 0:
    logger.info("Use no data augmentation")
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
elif args.aug == 1:
    logger.info("Use data augmentation 1: RandomGrayscale")
    train_transform = transforms.Compose([transforms.RandomGrayscale(p=0.2),  # 0.2的概率使用灰度图像
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(),  # 随机遮挡图像的一部分
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
elif args.aug == 2:
    logger.info("Use data augmentation 2: RandomCrop + RandomCrop")
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),  # 先四周填充0，再将图像随机裁剪成32*32
                                          transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# create dataset
trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT,
                                        train=True,
                                        download=True,
                                        transform=train_transform)
testset = torchvision.datasets.CIFAR10(root=DATA_ROOT,
                                       train=False,
                                       download=True,
                                       transform=test_transform)

# create dataLoder
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=4)
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=4)

# 2 MODEL
# activation function
activations = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "rrelu": nn.RReLU,
    "sigmoid": nn.Sigmoid,
    'tanh': nn.Tanh
}
if args.activation not in activations.keys():
    args.activation = "relu"
    logger.warning("Illegal activation: {}, activation should be in {}, reset to default \"relu\""
                   .format(args.activation, list(activations.keys())))
activation = activations[args.activation]
logger.info("Use activation: {}".format(args.activation))

# 设置是否使用多一层隐藏层+dropout
hidden_dim = args.hidden
if hidden_dim:
    logger.info("Add hidden layer, hidden dim is {}".format(hidden_dim))
    if args.dropout:
        dropout_p = args.dropout
        if dropout_p >= 1 or dropout_p <= 0:
            logger.warning("dropout prob p={} should satisfy 0<p<1, reset to default 0.2".format(dropout_p))
            dropout_p = 0.2
        logger.info("Add dropout to hidden layer, p={}".format(dropout_p))
    else:
        logger.info("Do not add dropout to hidden layer")
        dropout_p = None
else:
    dropout_p = None
    logger.info("No hidden layer")

models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2
}
if args.model_name not in models.keys():
    args.model_name = "resnet34"
    logger.warning("Illegal model_name: {}, model_name should be in {}, reset to default \"resnet34\""
                   .format(args.model_name, list(models.keys())))
net = models[args.model_name](pretrained=False, progress=True, activation=activation, hidden_dim=hidden_dim,
                              dropout_p=dropout_p, num_classes=10, zero_init_residual=True)
net.to(device)
logger.info("Use model: {}".format(args.model_name))

# 3 LOSS FUNCTION
criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for classification task

# 4 OPTIMIZER
# 选择优化器
optimizers = {
    'sgd': lambda n: optim.SGD(params=n.parameters(), lr=LR, momentum=0.9, weight_decay=L2_REG),
    'adagrad': lambda n: optim.Adagrad(params=n.parameters(), lr=LR, weight_decay=L2_REG),
    'rmsprop': lambda n: optim.RMSprop(params=n.parameters(), lr=0.01, momentum=0.9, weight_decay=L2_REG),
    'adadelta': lambda n: optim.Adadelta(params=n.parameters(), lr=LR, weight_decay=L2_REG),
    'adam': lambda n: optim.Adam(params=n.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_REG)
}
if args.optim not in optimizers.keys():
    args.optim = "sgd"
    logger.warning("Illegal optim: {}, optim should be in {}, reset to default \"sgd\""
                   .format(args.optim, list(optimizers.keys())))
optimizer = optimizers[args.optim](net)
logger.info("Use optimizer: {}".format(args.optim))

# 设置学习率调整策略
schedulers = {
    'steplr': lambda o: optim.lr_scheduler.StepLR(optimizer=o, step_size=30, gamma=0.1),
    'multisteplr': lambda o: optim.lr_scheduler.MultiStepLR(optimizer=o, milestones=[135, 185, 240], gamma=0.1),
    'explr': lambda o: optim.lr_scheduler.ExponentialLR(optimizer=o, gamma=0.9, last_epoch=-1),
    'coslr': lambda o: optim.lr_scheduler.CosineAnnealingLR(optimizer=o, T_max=10, eta_min=0)
}
if args.scheduler not in schedulers.keys():
    args.scheduler = "multisteplr"
    logger.warning("Illegal scheduler: {}, scheduler should be in {}, reset to default \"multisteplr\""
                   .format(args.scheduler, list(schedulers.keys())))
scheduler = schedulers[args.scheduler](optimizer)
logger.info("Use scheduler: {}".format(args.scheduler))


# 5 TRAINING
def train():
    logger.info('Training start!')
    start = time.time()
    epoch_start = time.time()

    train_losscurve = list()
    valid_losscurve = list()
    train_acc_curve = list()
    valid_acc_curve = list()
    max_acc = 0.

    # 构建 SummaryWriter
    if args.save_board:
        logger.info("Save tensorboard info to {}".format(args.boardf))
        writer = SummaryWriter(args.boardf)

    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()  # 切换到训练模式
        for i, data in enumerate(trainloader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # 打印训练信息，记录训练损失、准确率
            loss_mean += loss.item()
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                train_losscurve.append(round(loss_mean, 4))
                train_acc_curve.append(round(correct / total, 4))
                logger.info("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}"
                            .format(epoch + 1, MAX_EPOCH, i + 1, len(trainloader), loss_mean, correct / total))
                loss_mean = 0.

        # 每个epoch，记录weight, bias的grad, data
        if args.save_board:
            for name, param in net.named_parameters():
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()  # 切换到评估模式
            with torch.no_grad():
                for j, data in enumerate(testloader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                    loss_val += loss.item()

                acc = correct_val / total_val
                if acc > max_acc:  # 更新最大正确率
                    max_acc = acc
                valid_losscurve.append(round(loss_val / len(testloader), 4))  # 记录损失函数曲线
                valid_acc_curve.append(round(correct_val / total_val, 4))  # 记录准确率曲线
                logger.info("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}"
                            .format(epoch + 1, MAX_EPOCH, j + 1, len(testloader), loss_val / len(testloader),
                                    correct_val / total_val))

        epoch_end = time.time()
        logger.info('Time:\t{}'.format(round(epoch_end - epoch_start, 2)))
        epoch_start = time.time()

    logger.info('Training finished, {} epochs take {} sec'.format(MAX_EPOCH, round(time.time() - start)))
    logger.info('Max validation accuracy is: {:.2%}'.format(max_acc))
    return train_losscurve, train_acc_curve, valid_losscurve, valid_acc_curve, max_acc


# 6 SAVE MODEL
def save_model(train_losscurve, train_acc_curve, valid_losscurve, valid_acc_curve, max_acc):
    current_time = time.strftime("%Y_%m_%d_%H.%M.%S", time.localtime())

    # 保存模型参数
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.save(net.state_dict(), os.path.join(model_path, args.model_name + '_' + current_time + '.pth'))

    # 保存四条曲线以便可视化
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    with open(os.path.join(args.outf, args.outfname + args.model_name + "_" + current_time + '.txt', 'w')) as f:
        f.write('Training loss:\n')
        f.write(str(train_losscurve))
        f.write('\n\nTraining acc:\n')
        f.write(str(train_acc_curve))
        f.write('\n\nValidation loss:\n')
        f.write(str(valid_losscurve))
        f.write('\n\nValidation acc:\n')
        f.write(str(valid_acc_curve))
        f.write('\n\nThe max validation accuracy is: {:.2%}\n'.format(max_acc))


def main():
    train_result = train()
    save_model(*train_result)


if __name__ == "__main__":
    main()
