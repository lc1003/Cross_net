# -*- coding:utf-8 -*-

import tensorflow as tf
from config import get_parameter
import data_generator
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parameter = get_parameter()

data_path = parameter['data_set']
list_path = parameter['list']
result_path = parameter['result']

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


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op[0].shape[-1]
    # input_op = tf.convert_to_tensor(input_op)
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='VALID',
                            use_cudnn_on_gpu=None, data_format=None, name=None)
        bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float64)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)

    return activation


def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float64), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)

    return activation


def max_pool(input_op, name, kh, kw, dh, dw):
    result = tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='VALID', name=name)
    return result


def BN(x, out_size):
    fc_mean, fc_var = tf.nn.moments(x, axes=[0])
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    shift = tf.cast(shift, dtype=tf.float64)
    scale = tf.cast(scale, dtype=tf.float64)

    epsilon = 0.0001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_updata():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    mean, var = mean_var_with_updata()

    X = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

    return X


def cross(input1, input2):
    def reshape(input):
        shp = input.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        flattened_shape_sqrt = int(flattened_shape ** 0.5)
        reshape = tf.reshape(input, [-1, flattened_shape_sqrt, flattened_shape_sqrt], name="resh1")
        return reshape

    input1 = reshape(input1)
    input2 = reshape(input2)
    sigmoid = tf.nn.sigmoid(input2)
    tanh = tf.nn.tanh(input2)

    # data1 = tf.matmul(input1, sigmoid)
    # data2 = tf.matmul(tanh, sigmoid)
    data1 = input1 * sigmoid
    data2 = tanh * sigmoid

    data = data1 + data2

    return data


def inference_op(input_op, keep_prob=0.9):

    # block1 -- outputs
    RGB1_1 = conv_op(input_op['RGB'], name="RGB1_1", kh=3, kw=3, n_out=32, dh=2, dw=2)
    RGB1_1 = BN(RGB1_1, 32)
    RGB1_2 = conv_op(RGB1_1, name="RGB1_2", kh=3, kw=3, n_out=32, dh=2, dw=2)
    RGB1_2 = BN(RGB1_2, 32)
    RGB_pool1 = max_pool(RGB1_2, name="RGB_pool1", kh=2, kw=2, dh=2, dw=2)

    RGB2_1 = conv_op(RGB_pool1, name="RGB2_1", kh=3, kw=3, n_out=64, dh=2, dw=2)
    RGB2_1 = BN(RGB2_1, 64)
    RGB2_2 = conv_op(RGB2_1, name="RGB2_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    RGB2_2 = BN(RGB2_2, 64)

    # block2 -- outputs
    Grey1_1 = conv_op(input_op['Grey'], name="Grey1_1", kh=3, kw=3, n_out=32, dh=2, dw=2)
    Grey1_1 = BN(Grey1_1, 32)
    Grey1_2 = conv_op(Grey1_1, name="Grey1_2", kh=3, kw=3, n_out=32, dh=2, dw=2)
    Grey1_2 = BN(Grey1_2, 32)
    Grey_pool1 = max_pool(Grey1_2, name="Grey_pool1", kh=2, kw=2, dh=2, dw=2)

    Grey2_1 = conv_op(Grey_pool1, name="Grey2_1", kh=3, kw=3, n_out=64, dh=2, dw=2)
    Grey2_1 = BN(Grey2_1, 64)
    Grey2_2 = conv_op(Grey2_1, name="Grey2_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    Grey2_2 = BN(Grey2_2, 64)

    Grey_fc = cross(Grey2_2, RGB2_2)
    RGB_fc = cross(RGB2_2, Grey2_2)

    # flatten
    shp1 = Grey_fc.get_shape()
    flattened_shape = shp1[1].value * shp1[2].value
    Grey_fc = tf.reshape(Grey_fc, [-1, flattened_shape], name="Grey_fc")
    fc1_1 = fc_op(Grey_fc, name="fc1_1", n_out=32)
    fc1_1_drop = tf.nn.dropout(fc1_1, keep_prob, name="fc1_1_drop")
    fc1_2 = fc_op(fc1_1_drop, name="fc1_2", n_out=102)

    shp2 = RGB_fc.get_shape()
    flattened_shape = shp2[1].value * shp2[2].value
    RGB_fc = tf.reshape(RGB_fc, [-1, flattened_shape], name="RGB_fc")
    fc2_1 = fc_op(RGB_fc, name="fc2_1", n_out=32)
    fc2_1_drop = tf.nn.dropout(fc2_1, keep_prob, name="fc2_1_drop")
    fc2_2 = fc_op(fc2_1_drop, name="fc2_2", n_out=102)

    fc = fc1_2 + fc2_2
    logits = fc / 2

    return logits


def train(logits, labels, lr=learning_rate):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdadeltaOptimizer(lr).minimize(cost)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return optimizer, cost, accuracy


def get_result(kind='cv'):
    if kind == 'cv':
        cv_batch = data_generator.batch_generator(kind='cv')
        cv_batch['input']['RGB'] = np.array(cv_batch['input']['RGB'])
        cv_batch['input']['Grey'] = np.array(cv_batch['input']['Grey'])
        cv_batch['input']['Grey'] = np.reshape(cv_batch['input']['Grey'], (-1, img_row, img_col, 1))
        cv_pred = inference_op(input_op=cv_batch['input'], keep_prob=1)
        optimizer, cost, acc = train(logits=cv_pred, labels=cv_batch['label'])
    elif kind == 'train':
        train_batch = data_generator.batch_generator(kind='train')
        train_batch['input']['RGB'] = np.array(train_batch['input']['RGB'])
        train_batch['input']['Grey'] = np.array(train_batch['input']['Grey'])
        train_batch['input']['Grey'] = np.reshape(train_batch['input']['Grey'], (batch_size, img_row, img_col, 1))
        pred = inference_op(input_op=train_batch['input'], keep_prob=keep_prob)
        optimizer, cost, acc = train(logits=pred, labels=train_batch['label'])
    else:
        test_batch = data_generator.batch_generator(kind='test')
        test_batch['input']['RGB'] = np.array(test_batch['input']['RGB'])
        test_batch['input']['Grey'] = np.array(test_batch['input']['Grey'])
        test_batch['input']['Grey'] = np.reshape(test_batch['input']['Grey'], (-1, img_row, img_col, 1))
        test_pred = inference_op(input_op=test_batch['input'], keep_prob=1)
        optimizer, cost, acc = train(logits=test_pred, labels=test_batch['label'])

    return optimizer, cost, acc


def net_train():

    optimizer, cost, accuracy = get_result(kind='train')
    cv_optimizer, cv_cost, cv_acc = get_result(kind='cv')
    test_optimizer, test_cost, test_acc = get_result(kind='test')

    initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # global_ = tf.Variable(tf.constant(0))
    # d = tf.train.exponential_decay(0.01, global_, decay_steps, decay_rate, staircase=False)

    saver = tf.train.Saver()
    if not os.path.exists('model/'):
        os.mkdir('model/')

    with tf.Session() as sess:

        if os.path.exists('model/checkpoint'):
            saver.restore(sess, 'model/model.ckpt')
        else:
            sess.run(initop)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        Accuracy = []
        Loss = []
        step = 0
        for i in range(iteration):
            step += 1
            _, loss, acc = sess.run([optimizer, cost, accuracy])
            Accuracy.append(cv_acc)
            Loss.append(cv_cost)

            if step % display_step == 0:
                print('----------------loss= %.5f   ----------------acc= %.5f' % (loss, acc))
            if step % 100 == 0:
                _, cv_loss, cv_accuracy = sess.run([cv_optimizer, cv_cost, cv_acc])
                print("Cross_validation acc = %.5f " % cv_accuracy)

        _, test_loss, test_accuracy = sess.run([test_optimizer, test_cost, test_acc])
        print("Test acc = %.5f" % test_accuracy)
        saver.save(sess, 'model/model.ckpt')
        print("training finish!")
        print("Test Finish!")
        with open(result_path + 'Accuracy_cross.txt', 'w') as accuracy_file:
            for item in range(len(Accuracy)):
                accuracy_file.write(Accuracy)
        with open(result_path + 'Loss_cross.txt', 'W') as loss_file:
            for item in range(len(Loss)):
                loss_file.write(Loss)


if __name__ == "__main__":
    net_train()