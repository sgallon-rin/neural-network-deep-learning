import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
import time
from tqdm import tqdm
from IPython import display
from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from loaders import get_cifar_loader
from logger_utils import logger
from config import HOME

mpl.use('Agg')

# ## Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 4
batch_size = 128

# add our package dir to path
# module_path = os.path.dirname(os.getcwd())
# home_path = module_path
home_path = HOME
# figures_path = os.path.join(home_path, 'reports', 'figures')
# models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
logger.info('Use device: {}'.format(device))
if device != torch.device("cpu"):
    logger.info("Device name: {}".format(torch.cuda.get_device_name(device)))

# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)


# for X, y in train_loader:
#     ## --------------------
#     # Add code as needed
#     print(X[0])
#     print(y[0])
#     print(X[0].shape)
#     img = np.transpose(X[0], [1, 2, 0])
#     plt.imshow(img * 0.5 + 0.5)
#     # plt.savefig('sample.png')
#     print(X[0].max())
#     print(X[0].min())
#     ## --------------------
#     break
# sys.exit(-1)


# This function is used to calculate the accuracy of model classification
def get_accuracy():
    pass


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device=torch.device('cpu')):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != torch.device('cpu'):
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    loss_curve = [0]  # loss
    # train_accuracy_curve = [[np.nan] * epochs_n]
    train_accuracy_curve = []
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    interval = batches_n // 10  # interval to show info
    loss_list = []  # use this to record the loss value of each step
    grads_list = []  # use this to record the loss gradient of each step
    grads_diff_list = []
    last_grad = 0  # last gradient
    index = 0  # index of loss, acc record
    logger.info('Training start!')
    start_time = time.time()
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            loss_curve[index] += loss.item()
            grad = model.classifier[4].weight.grad.clone()
            grads_diff_list.append(round(torch.norm(grad - last_grad).item(), 3))
            last_grad = grad

            # calculate correct samples and training accuracy
            _, predicted = torch.max(prediction.data, 1)
            total += y.size(0)
            correct += (predicted == y).squeeze().sum().cpu().numpy()

            # record loss and grad and accuracy 10 times every epoch
            if (i + 1) % interval == 0:
                loss_curve[index] /= interval
                acc = round(correct / total, 4)
                train_accuracy_curve.append(acc)  # add acc
                correct = 0
                total = 0
                logger.info("Train:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, epochs_n, i + 1, len(train_loader), loss_curve[index], acc))
                index += 1
                loss_curve.append(0)

        # losses_list.append(loss_list)
        # grads.append(grad)
        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 3))
        #
        # learning_curve[epoch] /= batches_n
        # axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        correct_val = 0
        total_val = 0
        loss_val = 0.
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                loss_val += loss.item()

            acc = correct_val / total_val
            if acc > max_val_accuracy:
                max_val_accuracy = acc
                max_val_accuracy_epoch = epoch
            val_accuracy_curve[epoch] = round(acc, 4)
            logger.info("\nValid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                epoch + 1, epochs_n, j + 1, len(val_loader), loss_val / len(val_loader), correct_val / total_val))

    logger.info('Training finished, {} epochs take {} sec'.format(epochs_n, round(time.time() - start_time)))
    logger.info('Max validation accuracy is: {:.2%}, reached at {}-th epoch.'
                .format(max_val_accuracy, max_val_accuracy_epoch + 1))

    return loss_curve, grads_diff_list, train_accuracy_curve, val_accuracy_curve


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    pass


def main():
    # Train your model
    # feel free to modify
    epo = 20
    loss_save_path = os.path.join(home_path, 'loss_landscape')
    grad_save_path = os.path.join(home_path, 'grad_landscape')
    acc_save_path = os.path.join(home_path, 'acc_landscape')
    if not os.path.exists(loss_save_path):
        os.mkdir(loss_save_path)
    if not os.path.exists(grad_save_path):
        os.mkdir(grad_save_path)
    if not os.path.exists(acc_save_path):
        os.mkdir(acc_save_path)
    logger.info("Save path made!")

    lrs = [1e-3, 2e-3, 1e-4, 5e-4]

    set_random_seeds(seed_value=2020, device=device)
    for lr in lrs:
        logger.info("Training for lr={}".format(lr))
        model1 = VGG_A()
        model2 = VGG_A_BatchNorm()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
        optimizer2 = torch.optim.Adam(model1.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss1, grads_diff1, train_acc1, val_acc1 = train(model1, optimizer1, criterion, train_loader, val_loader,
                                                         epochs_n=epo)
        loss2, grads_diff2, train_acc2, val_acc2 = train(model2, optimizer2, criterion, train_loader, val_loader,
                                                         epochs_n=epo)
        # save loss and grads difference
        with open(os.path.join(loss_save_path, 'loss_{}.txt'.format(lr)), 'a') as f:
            f.write(str(loss1) + '\n')
        with open(os.path.join(loss_save_path, 'bn_loss_{}.txt'.format(lr)), 'a') as f:
            f.write(str(loss2) + '\n')

        with open(os.path.join(grad_save_path, 'grads_{}.txt'.format(lr)), 'a') as f:
            f.write(str(grads_diff1) + '\n')
        with open(os.path.join(grad_save_path, 'bn_grads_{}.txt'.format(lr)), 'a') as f:
            f.write(str(grads_diff2) + '\n')

        # save train, val accuracy
        with open(os.path.join(acc_save_path, 'acc_{}.txt'.format(lr)), 'a') as f:
            f.write('train_acc' + str(train_acc1) + '\n')
            f.write('val_acc' + str(val_acc1) + '\n')
            f.write('train_acc_bn' + str(train_acc2) + '\n')
            f.write('val_acc_bn' + str(val_acc2) + '\n')
        logger.info("Record files saved.")
    # np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
    # np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    # min_curve = []
    # max_curve = []


def test():
    model = VGG_A_BatchNorm()
    lr = 1e-3
    epo = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    _ = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)


if __name__ == "__main__":
    # main()
    test()
