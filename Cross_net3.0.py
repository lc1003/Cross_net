# -*- coding:utf-8 -*-

import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, concatenate, Add
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from keras.backend import constant
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from config import get_parameter
from get_data import *
from util import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

parameter = get_parameter()
data_path = parameter['data_set']
list_path = parameter['list']
result_path = parameter['result']
model_path = parameter['model_directory']

img_row = parameter['img_row']
img_col = parameter['img_col']
batch_size = parameter['batch_size']
class_num = parameter['class_num']

learning_rate = 0.01
keep_prob = 0.9
iteration = 5000
display_step = 5
global_steps = 1000  # 总的迭代次数
decay_steps = 100  # 衰减次数
decay_rate = 0.96  # 衰减率


class ModelFace():
    def __init__(self):
        self.nb_calsses = class_num
        self.model = self.build_model()

    def build_model(self):
        input_data = Input(shape=[224, 224, 3])
        # stream_1
        conv1_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(input_data)
        conv1_1 = BatchNormalization()(conv1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)

        # conv1_3 = Conv2D(filters=64, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal',
        #         #                  use_bias=True, activation='relu')(conv1_2)
        #         # conv1_3 = BatchNormalization()(conv1_3)

        pool1_1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv1_2)
        pool1_1 = Dropout(0.2)(pool1_1)

        conv1_4 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(pool1_1)
        conv1_4 = BatchNormalization()(conv1_4)

        conv1_5 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(conv1_4)
        conv1_5 = BatchNormalization()(conv1_5)

        # conv1_6 = Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal',
        #                  use_bias=True, activation='relu')(conv1_5)
        # conv1_6 = BatchNormalization()(conv1_6)

        pool1_2 = MaxPooling2D(pool_size=[2, 2])(conv1_5)
        pool1_2 = Dropout(0.2)(pool1_2)


        # stream_2
        conv2_1 = Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(input_data)
        conv2_1 = BatchNormalization()(conv2_1)

        conv2_2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[2, 2], padding='same', kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)

        # conv2_3 = Conv2D(filters=64, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal',
        #                  use_bias=True, activation='relu')(conv2_2)
        # conv2_3 = BatchNormalization()(conv2_3)

        pool2_1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv2_2)
        pool2_1 = Dropout(0.2)(pool2_1)

        conv2_4 = Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same',
                         kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(pool2_1)
        conv2_4 = BatchNormalization()(conv2_4)

        conv2_5 = Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], padding='same',
                         kernel_initializer='he_normal',
                         use_bias=True, activation='relu')(conv2_4)
        conv2_5 = BatchNormalization()(conv2_5)

        # conv2_6 = Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal',
        #                  use_bias=True, activation='relu')(conv2_5)
        # conv2_6 = BatchNormalization()(conv2_6)

        pool2_2 = MaxPooling2D(pool_size=[2, 2])(conv2_5)
        pool2_2 = Dropout(0.2)(pool2_2)

        output1 = concatenate([pool1_2, pool2_2], axis=2)
        # pool3 = GlobalAveragePooling2D()(conv6)


        # 1 x 1
        conv3_1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(output1)
        # conv3_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True,
        #                  activation='relu')(conv3_1)
        # pool3 = GlobalAveragePooling2D()(conv3_2)


        flatten = Flatten()(conv3_1)
        dense1_1 = Dense(units=256, activation='relu', use_bias=True, kernel_initializer='he_normal')(flatten)
        dense1_1 = Dropout(0.2)(dense1_1)
        dense1_2 = Dense(units=128, activation='relu', use_bias=True, kernel_initializer='he_normal')(dense1_1)
        dense1_2 = Dropout(rate=0.2)(dense1_2)
        dense1_3 = Dense(units=self.nb_calsses, use_bias=True, kernel_initializer='he_normal')(dense1_2)

        dense2_1 = Dense(units=256, activation='relu', use_bias=True, kernel_initializer='he_normal')(flatten)
        dense2_1 = Dropout(0.2)(dense2_1)
        dense2_2 = Dense(units=128, activation='relu', use_bias=True, kernel_initializer='he_normal')(dense2_1)
        dense2_2 = Dropout(rate=0.2)(dense2_2)
        dense2_3 = Dense(units=self.nb_calsses, use_bias=True, kernel_initializer='he_normal')(dense2_2)

        pred1 = Activation(activation='softmax')(dense2_3)
        pred2 = Activation(activation='softmax')(dense1_3)

        merge = Add()([pred1, pred2])
        model_data = Model(inputs=input_data, outputs=merge)
        # model_data.summary()
        return model_data

    def train(self, batch_size=128, nb_epoch=10, data_augmentation=False):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.9)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        datas, labels =get_data(train=True, test=False)
        k = 7281 // batch_size

        if not data_augmentation:
            self.model.fit(datas, labels, batch_size=batch_size, epochs=nb_epoch, verbose=1)
             # self.model.fit(data_images, data_labels, steps_per_epoch=k, epochs=nb_epoch, verbose=1)
        else:  # 进行数据的增广
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False
            )
            datagen.fit(datas)
            self.model.fit_generator(datas, labels)
        self.save_model()

    def save_model(self, filepath=model_path + 'myface_01.model.h5'):
        self.model.save(filepath=filepath)

    def load_mdoel(self, filepath=model_path + 'myface.model.h5'):
        self.model = load_model(filepath=filepath)

    def Evaluate(self, dataset):
        score = self.model.evaluate(dataset.valid_images, dataset.valid_labels, verbose=1)
        print("%s:%.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # class Confusion(keras.callbacks.Callback):
    #     def __init__(self, validation_data, interval=1):
    #         self.interval = interval
    #         self.x_val, self.y_val = validation_data
    #
    #     def on_epoch_end(self, epoch, logs={}):
    #         if epoch % self.interval == 0:
    #             pred_y = self.model.predict(self.x_val, verbose=0)
    #             true_label = np.argmax(self.y_val, axis=1)
    #             pred_label = np.argmax(pred_y, axis=1)
    #             classes = gd.get_classes()
    #             confusion_matx(true_label, pred_label, classes)
    #
    # X_test, y_test = gd.get_data(train=False, test=True)
    # Conf = Confusion(validation_data=(X_test, y_test))

    def CM(self, x_val, y_val):
        pred_y = self.model.predict(x_val, verbose=0)
        true_label = np.argmax(y_val, axis=1)
        pred_label = np.argmax(pred_y, axis=1)
        classes = get_classes()
        confusion_matx(true_label, pred_label, classes)


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

    md = ModelFace()
    md.train()
    tests, test_labels = get_data(train=False, test=True)
    md.CM(tests, test_labels)