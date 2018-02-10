import math
import tensorflow as tf


def pool(x, c):

    num_subblock    = c['num_subblock']
    sequence_length = c['sequence_length']
    filter_size     = 2#c['filter_size']

    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(
        x,
        ksize=[1, sequence_length - filter_size + num_subblock , 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")

    return pooled
    
