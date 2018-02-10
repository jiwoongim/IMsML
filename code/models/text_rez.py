"Code from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py"

import tensorflow as tf
import numpy as np

from model_kits.bn_layer import bn 
from model_kits.pool_layer import pool
from model_kits.conv_layer import conv2d
from model_kits.rez_block_layer import stack, block

from utils.config import Config
from utils.nn_utils import init_weights, activation_fn, _get_variable


class TextRez(object):
    """
    A RESIDUAL for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0 ):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.00005)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


	c = Config()
        c['bottleneck'], c['use_bias'], c['stride'], c['num_blocks'], c['num_subblock'] = \
							False, False, 1, 3, 2
        c['sequence_length'] = sequence_length
        c['fc_units_out'] = num_classes
        c['num_classes']  = num_classes 

        # Create a convolution + maxpool layer for each filter size
        outputs = []
        for i, filter_size in enumerate(filter_sizes):
    
   	    with tf.variable_scope("conv-maxpool-%s" % filter_size):
                c['ksize'] = filter_size
                x = self._residual_network1(self.embedded_chars_expanded, c, num_filters)
                x = pool(x,c)
                outputs.append(x)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) * embedding_size
        self.h_pool = tf.concat(3, outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print 'flatten filter size %d' % num_filters_total

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def _residual_network1(self, x, c, num_filters, atype='relu'):

        with tf.variable_scope('scale1'):
             c['conv_filters_out'] = num_filters
             c['block_filters_internal'] = num_filters
             c['stack_stride'] = 1
             x = conv2d(x, c)
             x = bn(x, c)
             x = activation_fn(x, atype)
             x = stack(x, c)

        #with tf.variable_scope('scale2'):
        #    c['block_filters_internal'] = num_filters*2
        #    c['stack_stride'] = 2
        #    x = stack(x, c)

        #with tf.variable_scope('scale3'):
        #    c['block_filters_internal'] = num_filters*4
        #    c['stack_stride'] = 2
        #    x = stack(x, c)

        #x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
        return x
