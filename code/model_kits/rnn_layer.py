import math
import tensorflow as tf

from utils.nn_utils import init_weights


class RNNLayer(object):
    def __init__(self, D, M, scope_name):
        ''' D - Dimension of the input 
            M - Dimention of the output
            scope_name - Name of the layer '''

        self.D = D 
        self.M = M 
        self.scope_name = scope_name

        self._initialize_weights()


    def __call__(self, X):
        if type(xs) != list: xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))

        with tf.variable_scope(self.scope):
            return self.fp(X)


    def _initialize_params(self):
        '''Initialize parameters in the layer'''

        with tf.variable_scope(self.scope_name):

            self.Wx = tf.get_variable("Wx", shape=[self.D, self.M], \
                    initializer=init_weights('xavier'))

            hid2hid = [self.M, self.M]
            self.Wh = tf.get_variable("Wx", shape=hid2hid, \
                    initializer=init_weights('identity')(hid2hid))

            self.hbias = tf.get_variable("hbias", shape=[self.M], \
                    initialzer=init_weights('zeros'))

            self.params = [self.Wx, self.Wh, self.hbias]


    def copy(self, scope=None):
        scope = self.scope_name + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(v.name, v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)


    def fp(self, X, H):
        '''Forward propagation'''

        with tf.variable_scope(self.scope_name):

            logit = tf.matmul(X, self.Wx) + tf.matmul(X, self.Wx) + self.hbias
            return activation(logit)



