# -*- coding:utf-8 -*-
from __future__ import print_function
import os

parameter = dict()

parameter['img_row'] = 224
parameter['img_col'] = 224
parameter['class_num'] = 102
parameter['batch_size'] = 102
parameter['iteration'] = 50000
parameter['learning_rate'] = 0.01

parameter['data_set'] = './data_set/'
parameter['list'] = 'list/'
parameter['model_directory'] = './model/'
parameter['result'] = './result/'
parameter['data_aug'] = './data_aug/'


def get_parameter():
    set_parameter()
    return parameter


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


def set_parameter():    # 创建目录
    # if not os.path.exists(parameter['root_path']+'dataset'):
    #     os.mkdir('dataset')
    # if not os.path.exists(parameter['file_directory']):
    #     os.mkdir('dataset/UCF101')
    if not os.path.exists(parameter['list']):
        os.mkdir(parameter['list'])
    if not os.path.exists(parameter['model_directory']):
        os.mkdir(parameter['model_directory'])
    if not os.path.exists(parameter['data_set']):
        os.mkdir(parameter['data_set'])
    if not os.path.exists(parameter['data_aug']):
        os.mkdir(parameter['data_aug'])
    # if not os.path.exists(parameter['model_directory']):
    #     os.mkdir(parameter['index_directory'])
    # if not os.path.exists(parameter['model_directory']):
    #     os.mkdir(parameter['model_directory'])
    # if not os.path.exists(parameter['img_directory']):
    #     os.mkdir(parameter['img_directory'])
    # if not os.path.exists(parameter['opt_flow_directory']):
    #     os.mkdir(parameter['opt_flow_directory'])


if __name__ == "__main__":
     set_parameter()
