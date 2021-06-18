#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2021/6/18 13:54
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : eval_EAST.py
# @Description : evaluate EAST
# In this project, test set is not labeled,
# so just detect word position in test set and save them for recognition task.


import time
import torch
import subprocess
import os
from EAST.model import EAST
from detect_EAST import detect_dataset
import shutil
from config import HOME, DATA_ROOT
from logger_utils import logger


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_name))
    logger.info("Loaded EAST model parameters from: {}".format(model_name))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    # os.chdir(submit_path)
    # res = subprocess.getoutput('zip -q submit.zip *.txt')
    # res = subprocess.getoutput('mv submit.zip ../')
    # os.chdir('../')
    # res = subprocess.getoutput('python ./EAST/evaluate/script.py –g=./EAST/evaluate/gt.zip –s=./submit.zip')
    # print(res)
    # os.remove('./submit.zip')
    logger.info('eval time is {}'.format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


if __name__ == '__main__':
    model_name = os.path.join(HOME, "pths", "EAST", 'model_epoch_600.pth')
    test_img_path = os.path.join(DATA_ROOT, "test", "img")
    submit_path = os.path.join(DATA_ROOT, "test", "EAST_gt")
    eval_model(model_name, test_img_path, submit_path)
