import math
import tensorflow as tf 
from model_kits.ln_gru_layer import GRU_LN_Layer


class BiGRU_LN_Layer(object):

    def __init__(self, D, M, layer):
        ''' D - Dimension of the input 
            M - Dimention of the output
            scope_name - Name of the layer '''

        self.D = D 
        self.M = M 
        self.layer = layer

        self.fgru = GRU_LN_Layer(D, M / 2, layer+'_forward')
        self.bgru = GRU_LN_Layer(D, M / 2, layer+'_backward')
        self.bigru = [self.fgru, self.bgru]
        self.params = self.fgru.params + self.bgru.params


    def __call__(self, X):
        return self.propagate(X)


    def propagate(self, X, n_steps=5., h0=None):
        '''Forward propagation'''
        if h0 is None: h0 = self.h0 

        Xft, Xbt = X, tf.reverse(X, dims = [False, False, True])
        hf0, hb0 = h0[:,:self.M/2], h0[:,self.M/2:]
        fh_t1 = self.fgru.propagate(Xft, h0=hf0, n_steps=n_steps) 
        bh_t1 = self.bgru.propagate(Xbt, h0=hb0, n_steps=n_steps) 

        return tf.concat(2, [fh_t1, bh_t1])



