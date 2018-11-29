# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from config import get_parameter

parameter = get_parameter()

data_path = parameter['data_set']
list_path = parameter['list']

# get class
classes = []
if os.path.exists(data_path):
    if os.path.isdir(data_path):
        with open(list_path + 'classInd.txt', 'w') as classInd:
            classes = os.listdir(data_path)
            for i, item in enumerate(classes):
                if i < len(classes)-1:
                    classInd.write('%d %s \n' % (i+1, item))
                else:
                    classInd.write('%d %s' % (i+1, item))


train_list = []
test_list = []
cv_list = []
for class_name in classes:
    class_dir = data_path + '/' + class_name
    img_list = os.listdir(class_dir)
    index = 0
    for img in img_list:
        index += 1
        if index <= int(len(img_list)*0.8):
            train_list.append(class_name + ' ' + img)
        elif index <= int(len(img_list)*0.9):
            cv_list.append(class_name + ' ' + img)
        else:
            test_list.append(class_name + ' ' + img)


with open(list_path+'train_list.txt', 'w') as Train_list:
    for item in train_list:
        Train_list.write(item + '\n')

with open(list_path+'test_list.txt', 'w') as Test_list:
    for item in test_list:
        Test_list.write(item + '\n')

with open(list_path + 'cv_list.txt', 'w') as CV_list:
    for item in cv_list:
        CV_list.write(item + '\n')


traindata_list = []
testdata_list = []
cv_data_list = []
for class_name in classes:
    class_dir = data_path + '/' + class_name
    img_list = os.listdir(class_dir)
    index = 0
    for img in img_list:
        index += 1
        if index <= int(len(img_list)*0.8):
            item = ['train', class_name, img]
            traindata_list.append(item)
        elif index <= int(len(img_list)*0.9):
            item = ['cv', class_name, img]
            cv_data_list.append(item)
        else:
            item = ['test', class_name, img]
            testdata_list.append(item)

data_list = traindata_list + cv_data_list + testdata_list
name = ['data', 'class_name', 'img']
writerCSV = pd.DataFrame(columns=name, data=data_list)
writerCSV.to_csv('./list/data_list.csv', encoding='utf-8', index=False, header=False)