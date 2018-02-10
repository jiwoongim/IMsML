import math
import tensorflow as tf

from utils.utils import base_name
from utils.nn_utils import init_weights, activation_fn

class Layer(object):
    def __init__(self, D, M, atype, scope_name):
        ''' D - Dimension of the input 
            M - Dimention of the output
            scope_name - Name of the layer '''

        self.D = D 
        self.M = M 
        self.atype = atype
        self.scope_name = scope_name

        self._initialize_params()


    def __call__(self, X):
        if type(xs) != list: xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))

        with tf.variable_scope(self.scope):     
            
            return self.fp(X)


    def _initialize_params(self):
        '''Initialize parameters in the layer'''

        with tf.variable_scope(self.scope_name) as sc:

            self.Wx = tf.get_variable("Wx", shape=[self.D, self.M], \
                    initializer=init_weights('xavier'))

            self.hbias = tf.get_variable("hbias", shape=[self.M], \
                    initializer=init_weights('zeros'))

            self.params = [self.Wx, self.hbias]

    
    def clone(self, scope_name=None):
        '''Duplicating the layer object'''

        if scope_name is None: scope_name = self.scope_name + "_clone"

        with tf.variable_scope(scope_name) as sc:
            for v in self.params:
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: \
                                            v.initialized_value())
            sc.reuse_variables()
            return Layer(self.D, self.M, self.atype, scope_name=sc)


    def fp(self, X, atype=None):
        '''Forward propagation'''

        if atype is None : atype = self.atype

        with tf.variable_scope(self.scope_name):

            logit = tf.matmul(X, self.Wx) + self.hbias
            return activation_fn(logit, atype)


    def get_logit(self, X):

        with tf.variable_scope(self.scope_name):

            return tf.matmul(X, self.Wx) + self.hbias

