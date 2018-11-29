# -*- coding:utf-8 -*-

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from config import get_parameter
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        self.nb_calsses = 10
        self.model = self.build_model()

    def build_model(self):
        input_data = Input(shape=[224, 224, 3])
        # stream_1
        conv1_1 = Conv2D(filters=16, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(input_data)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_2 = Conv2D(filters=16, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        pool1_1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv1_2)
        pool1_1 = Dropout(0.1)(pool1_1)
        conv1_3 = Conv2D(filters=32, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(pool1_1)
        conv1_3 = BatchNormalization()(conv1_3)
        conv1_4 = Conv2D(filters=32, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(conv1_3)
        conv1_4 = BatchNormalization()(conv1_4)
        pool1_2 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv1_4)
        pool1_2 = Dropout(0.1)(pool1_2)

        # stream_2
        conv2_1 = Conv2D(filters=16, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(input_data)
        conv2_1 = BatchNormalization()(conv2_1)
        conv2_2 = Conv2D(filters=16, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)
        pool2_1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv2_2)
        pool2_1 = Dropout(0.1)(pool2_1)
        conv2_3 = Conv2D(filters=32, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(pool2_1)
        conv2_3 = BatchNormalization()(conv2_3)
        conv2_4 = Conv2D(filters=32, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(conv2_3)
        conv2_4 = BatchNormalization()(conv2_4)
        pool2_2 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv2_4)
        pool2_2 = Dropout(0.1)(pool2_2)
        output1 = Concatenate([pool1_2, pool2_2], axis=3)
        # pool3 = GlobalAveragePooling2D()(conv6)

        # 1 x 1
        conv3_1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(output1)
        conv3_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=True,
                         activation='relu')(conv3_1)
        pool3 = GlobalAveragePooling2D()(conv3_2)

        dense1 = Dense(units=128, activation='relu', use_bias=True, kernel_initializer='he_normal')(pool3)
        dense1 = Dropout(0.1)(dense1)
        dense2 = Dense(units=256, activation='relu', use_bias=True, kernel_initializer='he_normal')(dense1)
        dense2 = Dropout(rate=0.2)(dense2)
        dense3 = Dense(units=self.nb_calsses, use_bias=True, kernel_initializer='he_normal')(dense2)
        pred = Activation(activation='softmax')(dense3)

        model_data = Model(inputs=input_data, outputs=pred)
        # model_data.summary()
        return model_data

    def train(self, batch_size=128, nb_epoch=1000, data_augmentation=False):

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.9)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        data_gen = get_batch(kind='train', batch_size=batch_size)
        k = 32739 // batch_size

        if not data_augmentation:
            self.model.fit_generator(data_gen, steps_per_epoch=k, epochs=nb_epoch, verbose=1)
        else:   # 进行数据的增广
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
            datagen.fit(data_gen['input'])
            self.model.fit_generator(data_gen['input'], data_gen['label'])
        self.save_model()

    def save_model(self, filepath=model_path + 'myface.model.h5'):
        self.model.save(filepath=filepath)

    def load_mdoel(self, filepath=model_path + 'myface.model.h5'):
        self.model = load_model(filepath=filepath)

    def Evaluate(self, dataset):
        score = self.model.evaluate(dataset.valid_images, dataset.valid_labels, verbose=1)
        print("%s:%.2f%%" % (self.model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

    md = ModelFace()
    md.train()