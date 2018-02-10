import math
import numpy as np
import tensorflow as tf


def init_weights(init_type):

    if init_type == 'xavier':
        return tf.contrib.layers.xavier_initializer()

    elif init_type == 'gaussian':
        return tf.random_normal_initializer()

    elif init_type == 'zeros':
        return tf.constant_initializer(0.0)

    elif init_type == 'ones':
        return tf.constant_initializer(1.0)


    #TODO : initialize the parameters as identity 
    elif init_type =='identity':

        def _initializer(shape, dtype=tf.float32):
            if len(shape) == 1:
                return tf.constant_op.constant(0., dtype=dtype, shape=shape)
            elif len(shape) == 2 and shape[0] == shape[1]:
                return tf.constant_op.constant(np.identity(shape[0], dtype))
            elif len(shape) == 4 and shape[2] == shape[3]:
                array = np.zeros(shape, dtype=float)
                cx, cy = shape[0]/2, shape[1]/2
                for i in range(shape[2]):
                    array[cx, cy, i, i] = 1
                return tf.constant_op.constant(array, dtype=dtype)
            else:
                raise
    
        return _initializer


def activation_fn(x, atype):
    
    if atype == 'relu':
        return tf.nn.relu(x)
    elif atype == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif atype == 'tanh':
        return tf.nn.tanh(x)
    elif atype == 'linear':
        return x
    elif atype == 'softmax':
        return tf.nn.softmax(x)



def linear_annealing(n, total, p_initial, p_final):
    """Linear annealing between p_initial and p_final
    over total steps - computes value at step n"""
    if n >= total:
        return p_final
    else:
        return p_initial - (n * (p_initial - p_final)) / (total)




def _get_variable(name,
                  shape,
                  initializer,
                  variable_name,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, variable_name]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)

