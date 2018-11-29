# -*- coding:utf-8 -*-

import shutil
import numpy as np
import os
import random
import tensorflow
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import itertools
import matplotlib.pyplot as plt
from config import get_parameter

parameter = get_parameter()

data_path = parameter['data_set']
list_path = parameter['list']
model_path = parameter['model_directory']

img_row = parameter['img_row']
img_col = parameter['img_col']
class_num = parameter['class_num']
learning_rate = parameter['learning_rate']


def save_best(best, acc, fold):
    if not os.path.exists(fold):
        os.mkdir(fold)
    filename = model_path + 'myface.model.h5'

    if acc >= best:
        shutil.copyfile(filename, fold + os.sep + 'myface.model.h5')


def adjust_learning_rate(optimizer, epoch):  # 学习率衰退
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def confusion_matx(true_label, pred, classes):    # classes 种类列表（字典形式）
    # lmr_matrix = confusion_matrix(true_label, pred_label)
    # plt.imshow(lmr_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('confusion matrix')
    # plt.colorbar()   # 画色彩渐变条
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # plt.xlabel('Pre label')
    # plt.ylabel('True label')
    # lmr_matrix = lmr_matrix.astype('float') / lmr_matrix.sum(axis=1)[:, np.newaxis]
    # fmt = '.2f'
    # thresh = lmr_matrix.max() / 2.
    # for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):   # 生成笛卡尔积的列表
    #     plt.text(j, i, format(lmr_matrix[i, j], fmt), horizontalalignment="center",
    #              color="black" if lmr_matrix[i, j] > thresh else "red")
    # plt.tight_layout()   # 为了避免它，轴域的位置需要调整。
    # plt.show()
    lmr_matrix = confusion_matrix(true_label, pred)
    # acc_score = accuracy_score(true_label, pred)
    # roc_auc_score = roc_auc_score(true_label, pred)
    plt.imshow(lmr_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=15)
    plt.yticks(tick_marks, classes, rotation=15)
    plt.xlabel('Pre label')
    plt.ylabel('True label')
    lmr_matrix = lmr_matrix.astype('float') / lmr_matrix.sum(axis=1)[:, np.newaxis]
    fmt = '.2f'
    thresh = lmr_matrix.max() / 2.
    for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):
        plt.text(j, i, format(lmr_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if lmr_matrix[i, j] > thresh else "blue")
    plt.tight_layout()
    plt.show()

