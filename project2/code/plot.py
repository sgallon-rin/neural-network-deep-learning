#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2021/5/27 10:55
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : plot.py
# @Description : plot curves

import os
import matplotlib.pyplot as plt
import numpy as np
from config import HOME


def plot_cifar(save_filename="./plot.png"):
    file_path = os.path.join(HOME, "saved_curves")
    with open(os.path.join(file_path, 'resnet34_aug2_lr1e-2_conv1311_batch64_epoch100.txt'), 'r') as f:
        line = f.readline()
        while True:
            if not line:
                break
            elif line.startswith('Training loss'):
                training_loss = f.readline()
                training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
            elif line.startswith('Training acc'):
                training_acc = f.readline()
                training_acc = [float(acc) for acc in training_acc[1:-2].split(',')]
            elif line.startswith('Validation loss'):
                validation_loss = f.readline()
                validation_loss = [float(loss) for loss in validation_loss[1:-2].split(',')]
            elif line.startswith('Validation acc'):
                validation_acc = f.readline()
                validation_acc = [float(acc) for acc in validation_acc[1:-2].split(',')]
            line = f.readline()

    num_epoch = len(validation_acc)
    skip = len(training_loss) // num_epoch
    training_loss = training_loss[::skip]
    training_acc = training_acc[::skip]

    plt.figure(figsize=(12, 12))

    plt.subplot(211)
    m1 = plt.plot(list(range(1, num_epoch + 1)), training_loss)
    m2 = plt.plot(list(range(1, num_epoch + 1)), validation_loss)
    plt.title('Loss vs time')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(["Training loss", "Validation loss"], loc='upper right')

    plt.subplot(212)
    m3 = plt.plot(list(range(1, num_epoch + 1)), training_acc)
    m4 = plt.plot(list(range(1, num_epoch + 1)), validation_acc)
    plt.title('Accuracy vs time')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend(["Training accuracy", "Validation accuracy"], loc='lower right')

    plt.savefig(save_filename)
    plt.show()


def plot_loss_landscape(save_filename="./plot.png"):
    train_curves = []
    with open('dropout_loss.txt') as f:
        while True:
            training_loss = f.readline()
            if not training_loss:
                break
            training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
            train_curves.append(training_loss)

    bn_train_curves = []
    with open('bn_loss2.txt') as f:
        while True:
            training_loss = f.readline()
            if not training_loss:
                break
            training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
            bn_train_curves.append(training_loss)

    train_curves = np.array(train_curves)
    bn_train_curves = np.array(bn_train_curves)

    min_curve = train_curves.min(0)
    max_curve = train_curves.max(0)
    bn_min_curve = bn_train_curves.min(0)
    bn_max_curve = bn_train_curves.max(0)
    x = [i / 10 for i in range(len(max_curve))]

    plt.figure(figsize=(15, 12))
    plt.style.use('fivethirtyeight')

    plt.plot(x, min_curve, color="#59aa6c")
    plt.plot(x, max_curve, color="#59aa6c")
    plt.plot(x, bn_min_curve, color="#c44e52")
    plt.plot(x, bn_max_curve, color="#c44e52")

    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="#9fc8ac")
    p2 = plt.fill_between(x, bn_min_curve, bn_max_curve, facecolor="#d69ba1")

    plt.title('loss landscape')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend([p1, p2], ['Standard VGG + Dropout', 'Standard VGG + BatchNorm'])

    plt.savefig(save_filename)
    plt.show()


if __name__ == "__main__":
    # plot_cifar()
    plot_loss_landscape()
