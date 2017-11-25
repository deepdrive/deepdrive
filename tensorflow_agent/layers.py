from __future__ import division
import numpy as np
import tensorflow as tf


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0

    def convolve(i, k):
        return tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def conv2d(x, name, num_features, kernel_size, stride, group):
    input_features = x.get_shape()[3]
    w = tf.get_variable(name + "_W", [kernel_size, kernel_size, int(input_features) // group, num_features])
    b = tf.get_variable(name + "_b", [num_features])
    return conv(x, w, b, kernel_size, kernel_size, num_features, stride, stride, padding="SAME", group=group)


def linear(x, name, size):
    input_size = np.prod(list(map(int, x.get_shape()[1:])))
    x = tf.reshape(x, [-1, input_size])
    w = tf.get_variable(name + "_W", [input_size, size], initializer=tf.random_normal_initializer(0.0, 0.005))
    b = tf.get_variable(name + "_b", [size], initializer=tf.zeros_initializer)
    return tf.matmul(x, w) + b


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')


def lrn(x):
    return tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

