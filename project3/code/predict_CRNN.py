#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2021/6/22 10:14
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : predict_CRNN.py
# @Description : predict label on test set using CRNN, use position labels from EAST

import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import CRNN.model as crnn
import CRNN.utils as utils
from CRNN.dataset import get_dataset
import CRNN.function as function
import CRNN.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import sys
import os
import shutil
from PIL import Image
from torchvision import transforms
from config import HOME, DATA_ROOT
from logger_utils import logger


def parse_arg():
    parser = argparse.ArgumentParser(description="CRNN predict")
    parser.add_argument('--cfg', default=os.path.join(HOME, "code", "CRNN", "CRNN_config.yaml"),
                        help='experiment configuration filename', type=str)
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        default="/home/jupyter/sjl/neural-network-deep-learning/project3/pths/CRNN/OWN/crnn/2021-06-21-10-09/checkpoints/checkpoint_99_acc_0.5379.pth")
    parser.add_argument('--mode', default="test", type=str)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f)
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def recognition(config, img_path, model, converter, device):
    inp_h = config.MODEL.IMAGE_SIZE.H
    max_w = config.MODEL.IMAGE_SIZE.MAX_W
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    transformer = transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
        transforms.ToTensor(),
    ])
    normalization = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path, "r")
    img = img.convert("RGB")

    # ratio resize
    img_h, img_w = img.size
    resize_ratio = inp_h / float(img_h)
    after_w = int(resize_ratio * img_w)
    if after_w == 0: after_w = 1
    img = img.resize((after_w, inp_h), resample=Image.BICUBIC)
    # w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    # h, w = img.shape
    # img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = transforms.Pad(padding=(0, 0, max_w - after_w, 0), fill=1)(img)  # padding
    # convert to CNN shape
    img = transformer(img)  # C, H, W

    # deal with images with channels < 3
    c, h, w = img.shape
    if c < 3:
        n = 3 - c
        img_new = img
        for i in range(n):
            img_new = torch.cat((img_new, img), 0)
        img = img_new
    assert img.shape[0] == 3

    img = normalization(img)

    # img = img.astype(np.float32)
    # img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    # img = img.transpose([2, 0, 1])

    # img = torch.from_numpy(img)
    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('results: {0}'.format(sim_pred))

    return sim_pred


if __name__ == '__main__':
    config, args = parse_arg()
    logger.info("Config loaded")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info("Device is {}".format(device))

    model = crnn.get_crnn(config).to(device)
    model_state_file = args.checkpoint
    if model_state_file == '':
        logger.error(" => no checkpoint found")
        sys.exit(-1)
    checkpoint = torch.load(model_state_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Model created, checkpoint loaded")

    # converter
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)  # get corpus
    logger.info("Alphabet loaded")

    # get test data
    input_img_path = os.path.join(DATA_ROOT, "test", "text_img")  # extracted text imgs to load
    logger.info("Input img path is {}".format(input_img_path))
    output_txt_path = os.path.join(DATA_ROOT, "test", "txt")  # gt to save
    logger.info("Output txt path is {}".format(output_txt_path))
    output_text_file = os.path.join(output_txt_path, 'CRNN_predict_labels.txt')
    logger.info("Output text file is {}".format(output_text_file))
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write('')  # 将标注文件其中的内容清空
    logger.info("Output text file cleared")

    imgs = os.listdir(input_img_path)
    for idx, filename in enumerate(imgs):
        if not filename.endswith(".jpg"):
            logger.info("[{}/{}]Skip non-jpg file: {}".format(idx, len(imgs), filename))
            continue
        logger.info("[{}/{}]Processing picture: {}".format(idx, len(imgs), filename))

        img_path = os.path.join(input_img_path, filename)
        pred = recognition(config, img_path, model, converter, device)

        with open(output_text_file, 'a', encoding='utf-8') as f:
            f.write(filename + ',' + pred + '\n')

    logger.info("All done")
