import math
import tensorflow as tf

from utils.nn_utils import init_weights
from model_kits.lstm_layer import LSTMLayer


class BiLSTMLayer(object):

    def __init__(self, D, M, scope_name):
        ''' D - Dimension of the input 
            M - Dimention of the output
            scope_name - Name of the layer '''

        self.D = D 
        self.M = M 
        self.scope_name = scope_name

        self.flstm = LSTMLayer(D,M, scope_name+'_forward')
        self.blstm = LSTMLayer(D,M, scope_name+'_backward')
        self.bilstm = [self.flstm, self.blstm]
        self.params = self.flstm.params + self.blstm.params


    def __call__(self, X):
        if type(xs) != list: xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))

        with tf.variable_scope(self.scope):
            return self.fp(X)


    def copy(self, scope=None):
        scope = self.scope_name + "_copy"

        with tf.variable_scope(scope) as sc:
           
            #TODO In the future, when it is necessary
            pass


    def fp(self, Xft, Xbt, y_ft, y_bt):
        '''Forward propagation'''

        fOUT_t1, fy_t1 = self.flstm.fp(Xft, y_ft) 
        bOUT_t1, by_t1 = self.blstm.fp(Xbt, y_bt) 

        return tf.reshape(tf.pack(fOut_t1, bOUT_t1), [-1])


