import math
import tensorflow as tf

from model_kits.bn_layer import bn
from model_kits.conv_layer import conv2d

from utils.utils import base_name
from utils.nn_utils import init_weights, activation_fn

"This code is from https://github.com/ry/tensorflow-resnet/"

def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c, atype='relu'):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1
    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv2d(x, c)
            x = bn(x, c)
            x = activation_fn(x, atype)

        with tf.variable_scope('b'):
            x = conv2d(x, c)
            x = bn(x, c)
            x = activation_fn(x, atype)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv2d(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            x = conv2d(x, c)
            x = bn(x, c)
            x = activation_fn(x, atype)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            #assert c['stride'] == 1
            x = conv2d(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv2d(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation_fn(x + shortcut, atype)

