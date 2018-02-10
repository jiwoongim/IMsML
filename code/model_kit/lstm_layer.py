import math
import tensorflow as tf

from utils.nn_utils import init_weights



class LSTMLayer(object):

    def __init__(self, D, M, scope_name):
        ''' D - Dimension of the input 
            M - Dimention of the output
            scope_name - Name of the layer '''

        self.D = D 
        self.M = M 
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

        with tf.variable_scope(self.scope_name):

            self.Wx_x = tf.get_variable("Wx_x", shape=[self.D, self.M * 3], \
                                initializer=init_weights('xavier'))

            self.Wx_O = tf.get_variable("Wx_O", shape=[self.M, self.M * 3], \
                                initializer=init_weights('xavier'))

            #self.Wx_y = tf.get_variable("Wx_y", shape=[self.M, self.M * 3], \
            #                    initializer=init_weights('xavier'))

            self.b_in = tf.Variable(tf.concat(0, [\
                                tf.zeros([self.M * 2,], tf.float32), \
                                tf.ones([self.M,], tf.float32)]), "b_in")
 

            self.WIN_O = tf.get_variable("WIN_O", shape=[self.D, self.M], \
                                initializer=init_weights('xavier'))

            self.WIN_y = tf.get_variable("WIN_y", shape=[self.M, self.M], \
                                initializer=init_weights('xavier'))

            self.b_IN = tf.get_variable("b_IN", shape=[self.M], \
                    initializer=init_weights('zeros'))

            self.init_O = tf.get_variable("init_O", shape=[1, self.M], \
                    initializer=init_weights('zeros'))

            self.init_y = tf.get_variable("init_y", shape=[1, self.M], \
                    initializer=init_weights('zeros'))

            self.params = [ self.Wx_x, self.Wx_O, \
                            self.WIN_O, self.WIN_y, self.b_in, self.b_IN]


    def copy(self, scope=None):
        scope = self.scope_name + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(v.name, v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)


    def fp(self, Xt, H_t):
        '''Forward propagation'''

        with tf.variable_scope(self.scope_name):
    
            concat_pre_act = tf.nn.sigmoid(tf.matmul(Xt, self.Wx_x) +\
                                           tf.matmul(H_t, self.Wx_O) + self.b_in)
                                           #tf.matmul(y_t, self.Wx_y) + self.b_in)

            in_t1  = concat_pre_act[:, :self.M]
            out_t1 = concat_pre_act[:, self.M   : self.M*2]
            fg_t1  = concat_pre_act[:, self.M*2 : self.M*3]
            IN_t1  = tf.nn.tanh(    tf.matmul(Xt , self.WIN_O) + \
                                    tf.matmul(H_t, self.WIN_y) + self.b_IN )

            OUT_t1 = tf.nn.tanh(out_t1*fg_t1 + IN_t1*in_t1)
            y_t1   = OUT_t1*out_t1

            return OUT_t1, y_t1



