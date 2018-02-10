import math
import tensorflow as tf

from tensorflow.contrib import rnn
from utils.nn_utils import init_weights
from model_kit.lstm_layer import LSTMLayer


class BiLSTMLayerTF(object):

    def __init__(self, D, M, scope_name):
        ''' D - Dimension of the input 
            M - Dimention of the output
            scope_name - Name of the layer '''

        self.D = D 
        self.M = M 
        self.scope_name = scope_name

        # Define lstm cells with tensorflow
        with tf.variable_scope(self.scope_name+'_forward') as scope:     
            self.flstm = rnn.BasicLSTMCell(M, forget_bias=1.0)
        with tf.variable_scope(self.scope_name+'_forward') as scope:     
            self.blstm = rnn.BasicLSTMCell(M, forget_bias=1.0)

        self.bilstm = [self.flstm, self.blstm]
        self.params = self.flstm.variables + self.blstm.variables


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


    def fp(self, X):

        '''Forward propagation'''
        # Get lstm cell output
        with tf.variable_scope(self.scope_name) as scope:     
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(\
                                                    self.flstm, \
                                                    self.blstm, \
                                                    X,
                                                    dtype=tf.float32)
            except Exception: # Old TensorFlow version only returns outputs not states
                print 'Using Old version Tensorflow BiLSTM'
                outputs = rnn.static_bidirectional_rnn( self.flstm, \
                                                        self.blstm, \
                                                        X, \
                                                        dtype=tf.float32)
            return outputs





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



