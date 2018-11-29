# -*- coding:utf-8 -*-

from config import get_parameter
from PIL import Image, ImageEnhance
from glob import glob
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import os
import cv2
from skimage.util import random_noise
from skimage import exposure

parameter = get_parameter()
data_path = parameter['data_set']
list_path = parameter['list']
data_aug_path = parameter['data_aug']


# 加高斯噪声
def addNoise(img):

    return random_noise(img, mode='gaussian', seed=13, clip=True) * 255


# 调节亮度
def light_change(img):

    scaler = random.uniform(0.5, 1.5)
    img = exposure.adjust_gamma(img, scaler)

    return img


# 翻转
def img_filp(img):

    img1 = img.transpose(Image.FLIP_LEFT_RIGHT)   # 左右翻转
    img2 = img.transpose(Image.FLIP_TOP_BOTTOM)   # 上下翻转

    return img1, img2


# 改变对比度
def change_contrast(img, contrast):

    enh_con = ImageEnhance.Contrast(img)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)

    return image_contrasted


# 图片旋转；旋转时候背景填充为黑色
def img_rotate(img, angel=30):

    img = img.rotate(angel)

    return img


# 图像锐度的变化
def sharpness_change(img, sharpness=1.5):

    enh_sha = ImageEnhance.Sharpness(img)
    image_sharped = enh_sha.enhance(sharpness)

    return image_sharped


def img_aug(img_dir, raw_path, aug_path):

    img = cv2.imread(raw_path + img_dir.split(' ')[0] + img_dir.split(' ')[-1])

    gau_image = addNoise(img)

    # light = light_change(img)
    #
    # flip_horizontal, flip_vertical = img_filp(img)
    #
    # img_contrast = change_contrast(img, contrast=2.0)
    #
    # rotate = img_rotate(img, angel=30)
    #
    # gau_add_light = light_change(gau_image)

    cv2.imwrite(aug_path + 'aug_' + img_dir.split(' ')[0] + '_' + img_dir.split(' ')[-1], gau_image)


def get_row_list():

    data_list = []
    with open(list_path + 'train_list.txt', 'r') as data_file:
        lines = data_file.readlines()
        for line in lines:
            line = line.strip('\n')
            data_list.append(line)

    return [item for item in data_list]


if __name__ == '__main__':

    row_img_path = get_row_list()

    for item in row_img_path:
        img_aug(item, data_path, data_aug_path)
