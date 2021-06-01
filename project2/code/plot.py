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

lrs = [1e-3, 2e-3, 1e-4, 5e-4]
LANDSCAPE_HOME = os.path.join(HOME, "landscape")


def plot_cifar(save_fig_name="plot.png"):
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
    plt.savefig(save_fig_name)
    plt.show()


def plot_loss_landscape(result_file, bn_result_file, save_fig_name="plot.png"):
    train_curves = []
    with open(result_file, 'r') as f:
        while True:
            training_loss = f.readline()
            if not training_loss:
                break
            training_loss = [float(loss) for loss in training_loss[1:-2].split(',')]
            train_curves.append(training_loss)

    bn_train_curves = []
    with open(bn_result_file, 'r') as f:
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
    plt.title('Loss Landscape')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend([p1, p2], ['Standard VGG', 'Standard VGG + BatchNorm'])
    plt.savefig(save_fig_name)
    plt.show()


def plot_grad_predictiveness(result_file, bn_result_file, save_fig_name="plot.png"):
    grads_diff = []
    with open(result_file, 'r') as f:
        while True:
            grad_diff = f.readline()
            if not grad_diff:
                break
            grad_diff = [float(loss) for loss in grad_diff[1:-2].split(',')]
            grads_diff.append(grad_diff)

    bn_grads_diff = []
    with open(bn_result_file, 'r') as f:
        while True:
            bn_grad_diff = f.readline()
            if not bn_grad_diff:
                break
            bn_grad_diff = [float(loss) for loss in bn_grad_diff[1:-2].split(',')]
            bn_grads_diff.append(bn_grad_diff)

    grads_diff = np.array(grads_diff)[:, 3:]
    bn_grads_diff = np.array(bn_grads_diff)[:, 3:]
    min_curve = grads_diff.min(0)
    max_curve = grads_diff.max(0)
    bn_min_curve = bn_grads_diff.min(0)
    bn_max_curve = bn_grads_diff.max(0)
    x = [i for i in range(len(max_curve))]

    plt.figure(figsize=(15, 12))
    plt.style.use('fivethirtyeight')
    plt.plot(x, min_curve, color="#59aa6c")
    plt.plot(x, max_curve, color="#59aa6c")
    plt.plot(x, bn_min_curve, color="#c44e52")
    plt.plot(x, bn_max_curve, color="#c44e52")
    p1 = plt.fill_between(x, min_curve, max_curve, facecolor="#9fc8ac")
    p2 = plt.fill_between(x, bn_min_curve, bn_max_curve, facecolor="#d69ba1")
    plt.title('Gradient Predictiveness')
    plt.ylabel('gradient difference')
    plt.xlabel('step')
    plt.legend([p1, p2], ['Standard VGG', 'Standard VGG + BatchNorm'])
    plt.savefig(save_fig_name)
    plt.show()


def plot_beta_smoothness(result_file, save_fig_name="plot.png"):
    with open(result_file, 'r') as f:
        beta_smooth = f.readline()
        beta_smooth_bn = f.readline()

    beta_smooth = [float(s) for s in beta_smooth[1:-2].split(',')][3:]
    beta_smooth_bn = [float(s) for s in beta_smooth_bn[1:-2].split(',')][3:]

    plt.figure(figsize=(15, 12))
    plt.style.use('fivethirtyeight')
    plt.plot(beta_smooth, color="#59aa6c")
    plt.plot(beta_smooth_bn, color="#c44e52")
    plt.xlabel('step')
    plt.ylabel('max difference in gradient')
    plt.title('\"Effective\" β-Smoothness')
    plt.legend(['Standard VGG', 'Standard VGG + BatchNorm'])
    plt.savefig(save_fig_name)
    plt.show()


if __name__ == "__main__":
    # plot_cifar()

    loss_f = os.path.join(LANDSCAPE_HOME, "loss.txt")
    bn_loss_f = os.path.join(LANDSCAPE_HOME, "bn_loss.txt")
    save_loss_f = os.path.join(LANDSCAPE_HOME, "loss_landscape.png")
    plot_loss_landscape(loss_f, bn_loss_f, save_loss_f)

    grads_diff_f = os.path.join(LANDSCAPE_HOME, "grads_diff.txt")
    bn_grads_diff_f = os.path.join(LANDSCAPE_HOME, "bn_grads_diff.txt")
    save_grads_diff_f = os.path.join(LANDSCAPE_HOME, "grad_predictiveness.png")
    plot_grad_predictiveness(grads_diff_f, bn_grads_diff_f, save_grads_diff_f)

    beta_f = os.path.join(LANDSCAPE_HOME, "beta_smoothness.txt")
    save_beta_f = os.path.join(LANDSCAPE_HOME, "beta_smoothness.png")
    plot_beta_smoothness(beta_f, save_beta_f)
