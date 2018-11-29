# -*- coding=utf-8 -*-

import numpy as np
import random
import csv
import cv2
import os
import gc
from tqdm import tqdm
import pandas as pd
from keras.utils import np_utils
# from torchvision import transforms as T
from config import get_parameter

parameter = get_parameter()

data_path = parameter['data_set']
list_path = parameter['list']
aug_path = parameter['data_aug']

img_row = parameter['img_row']
img_col = parameter['img_col']
batch_size = parameter['batch_size']
class_num = parameter['class_num']


def get_classes():
    classes = list()
    with open(list_path + 'classInd.txt', 'r') as class_list:
        lines = class_list.readlines()
        for item in lines:
            item = item.strip('\n').split(' ')[1]
            classes.append(str(item))
    return classes


def resize_image(image, height=img_col, width=img_row):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, channels = image.shape
    # print(h , w)
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width), interpolation=cv2.INTER_NEAREST)


def get_list(train=False, test=False):
    if train:
        with open(list_path + 'train_list.txt', 'r') as data_list:
            lines = data_list.readlines()
        return list(map(lambda x: x.strip('\n'), lines))
    elif test:
        with open(list_path + 'test_list.txt', 'r') as data_list:
            lines = data_list.readlines()
        return list(map(lambda x: x.strip('\n'), lines))
    else:
        with open(list_path + 'cv_list.txt', 'r') as data_list:
            lines = data_list.readlines()
        return list(map(lambda x: x.strip('\n'), lines))


def get_aug_list():
    aug_data = os.listdir(aug_path)
    for i in range(len(aug_data)):
        aug_data[i] = aug_data[i]
    return


def get_files(root, mode):
    # for test
    if mode == "test":
        files = []
        for dir in root:
            files.append(data_path + dir.split(' ')[0] + '/' + dir.split(' ')[-1])
        files = sorted(files)  # 对字典中的列表进行排序
        return files
    elif mode != "test":
        # for train and val
        data_list = list()
        for item in root:
            data_list.append(item.split(' ')[0] + '/' + item.split(' ')[-1])
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: data_path + x, data_list))
        # map():接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回
        # jpg_image_1 = list(map(lambda x: glob(x + "/*.jpg"), image_folders))   # jpg_image1 存储图片的路径
        # jpg_image_2 = list(map(lambda x: glob(x + "/*.JPG"), image_folders))
        # all_images = list(chain.from_iterable(jpg_image_1 + jpg_image_2))        # 将两个列表都连接起来
        print("loading train dataset")
        for file in tqdm(image_folders):
            all_data_path.append(file)
        #     labels.append(int(file.split("/")[-2]))  # 各类图片存放在各类的文件夹之下
        # all_files = pd.DataFrame({"filename": all_data_path, "label": labels})   # 创建一个由字典组成的表结构
        return all_data_path
    else:
        print("check the mode please!")


def get_data(train=False, test=False):

    if train:
        data_list = get_list(train=True)
        data_dir = get_files(data_list, mode='train')

    elif test:
        data_list = get_list(test=True)
        data_dir = get_files(data_list, mode='test')

    else:
        data_list = get_list()
        data_dir = get_files(data_list, mode='cv')

    # with open(data_list + 'classInd.txt', 'r') as class_list:
    #     lines = class_list.readlines()
    #     classes = list(map(lambda x: x.split(' ')[-1]))

    classes = get_classes()
    data_images = []
    data_labels = []
    for dir in data_dir:
        data_labels.append(float(classes.index(dir.split('/')[-2])))
        img = cv2.imread(dir)
        img = resize_image(img)
        img = np.array(img)
        image = img - np.mean(img)  # mean substraction
        image = image / 255.
        data_images.append(image)
    data_labels = np_utils.to_categorical(data_labels, class_num)

    data_images = np.array(data_images).reshape((-1, img_col, img_col, 3))
    data_labels = np.array(data_labels)
    # print(data_images)
    # print(data_labels)
    return data_images, data_labels
# def get_data(train=False, test=False):
#
#     if train:
#         data_list = get_list(train=True)
#         data_dir = get_files(data_list, mode='train')
#
#     elif test:
#         data_list = get_list(test=True)
#         data_dir = get_files(data_list, mode='test')
#
#     else:
#         data_list = get_list()
#         data_dir = get_files(data_list, mode='cv')
#
#     classes = get_classes()
#     data_num = len(data_dir)
#     images = np.zeros(shape=[data_num, img_row, img_col, 3], dtype=np.float32)
#     labels = np.zeros(shape=[data_num], dtype=np.int32)
#     for i, dir in enumerate(data_dir):
#         img = cv2.imread(dir)
#         img = resize_image(img)
#         img = np.array(img)
#         image = img - np.mean(img)  # mean substraction
#         image = image / 255.
#         images[i] = image
#         labels[i] = np_utils.to_categorical(float(classes.index(dir.split('/')[-2])), class_num)
#
#     images = np.array(images, dtype=np.float)
#     labels = np.array(labels, dtype=np.int)
#
#     return images, labels


def generater(train=False, test=False):
    while True:
        images, labels = get_data(train, test)
        yield (images, labels)


if __name__ == '__main__':
    get_data(train=True, test=False)
