import math
import tensorflow as tf

from utils.utils import base_name
from utils.nn_utils import init_weights, activation_fn, _get_variable

class Conv_Layer(object):
    def __init__(self, Xdim, kern_sz, filter_sz, atype, scope_name, \
                    stride=[2,2], padding='SAME', dformat='NHWC'):

        ''' 
            M - Dimention of the output
            N - Number of data
            C - Number of channels
            W - Width
            H - Height
            dformat - NCHW | NHWC 
            scope_name - Name of the layer '''

        self.Xdim       = Xdim
        self.atype      = atype
        self.padding    = padding
        self.scope_name = scope_name
        self.kern_sz    = kern_sz
        self.filter_sz  = filter_sz

        if data_format == 'NCHW':
            self.stride = [1, 1, stride[0], stride[1]]
            self.kernel_shape   = [ filter_sz[0], \
                                    filter_sz[1], \
                                    self.Xdim[1],   \
                                    kern_sz ]
        elif data_format == 'NHWC':
            self.stride = [1, stride[0], stride[1], 1]
            self.kernel_shape   = [ filter_sz[0], \
                                    filter_sz[1], \
                                    self.Xdim[-1],  \
                                    kern_sz ]

        self._initialize_params()


    def __call__(self, X):

        if type(xs) != list: xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))

        with tf.variable_scope(self.scope):     
            
            return self.fp(X)

      
    def _initialize_params(self, wtype='xavier'):
        '''Initialize parameters in the layer'''

        with tf.variable_scope(self.scope_name) as sc:

            self.Wx = tf.get_variable("Wx", shape=self.kernel_shape, \
                    initializer=init_weights(wtype))

            self.hbias = tf.get_variable("hbias", shape=[self.M], \
                    initializer=init_weights('zeros'))

            self.params = [self.Wx, self.hbias]


    def fp(self):

        conv    = tf.nn.conv2d(x, self.Wx, self.stride, self.padding, data_format=data_format)
        logit   = tf.nn.bias_add(conv, self.hbias, self.dformat)

        return activation_fn(logit, self.atype)


    def clone(self, scope_name=None):
        '''Duplicating the layer object'''

        if scope_name is None: scope_name = self.scope_name + "_clone"

        with tf.variable_scope(scope_name) as sc:
            for v in self.params:
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: \
                                            v.initialized_value())
            sc.reuse_variables()
            return Conv_Layer(self.D, self.M, self.atype, scope_name=sc)


def conv2d(x, c, wc=0.00005):
    ksize   = c['ksize']
    stride  = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, 1, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=0.1)

    collections = [tf.GraphKeys.VARIABLES, 'resnet_variables']

    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            variable_name='resnet_variables',
                            weight_decay=wc)

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


