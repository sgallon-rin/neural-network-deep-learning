#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2021/6/22 15:38
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : make_submit_gt.py
# @Description :

import os
from tqdm import tqdm
from config import HOME, DATA_ROOT
from logger_utils import logger

if __name__ == "__main__":
    submit_gt_path = os.path.join(DATA_ROOT, "test", "submit_gt")
    test_img_path = os.path.join(DATA_ROOT, "test", "img")
    EAST_label_txt_file = os.path.join(DATA_ROOT, "test", "txt", "test_own.txt")
    CRNN_label_txt_file = os.path.join(DATA_ROOT, "test", "txt", "CRNN_predict_labels.txt")

    if not os.path.exists(submit_gt_path):
        os.mkdir(submit_gt_path)
        logger.info("Output dir made")

    img_names = [f.split(".")[0] for f in os.listdir(test_img_path) if not f.startswith(".")]
    logger.info("Test set size: {}".format(len(img_names)))

    with open(EAST_label_txt_file, "r") as f:
        EAST_labels = f.readlines()
    with open(CRNN_label_txt_file, "r") as f:
        CRNN_labels = f.readlines()

    for img_name in tqdm(img_names):
        coords_line = [line.strip().split(",") for line in EAST_labels if img_name in line]
        text_line = [line.strip().split(",") for line in CRNN_labels if img_name in line]

        if not len(coords_line) == len(text_line):
            logger.warning("Label len mismatch! Skip img {}".format(img_name))
            continue
        elif len(coords_line) == 0:
            logger.warning("Empty label for img {}".format(img_name))

        gt_name = img_name + ".txt"
        with open(os.path.join(submit_gt_path, gt_name), "w") as f:
            for line in coords_line:
                sub_img_name = line[0]
                coords = line[1::]
                f.write(",".join(coords))
                matched = False
                text = ""
                for tline in text_line:
                    if tline[0] == sub_img_name:
                        matched = True
                        text = tline[1]
                        break
                if not matched:
                    logger.warning("EAST and CRNN label mismatch in mini-img {}".format(sub_img_name))
                f.write(",,{}\n".format(text))
