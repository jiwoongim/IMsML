import os, sys
import numpy as np

import tensorflow as tf

from utils.nn_utils import *
from utils.tf_utils import *
TINY    = 1e-5

class Layer_Norm(object):

    def __init__(self, D, M, name, numpy_rng):
        self.W       = initialize_weight(D, M,  name, numpy_rng, 'uniform') 
        self.eta         = theano.shared(np.ones((M,), dtype=theano.config.floatX), name='eta') 
        self.beta        = theano.shared(np.zeros((M,), dtype=theano.config.floatX), name='beta')

        self.params = [self.W, self.eta, self.beta]


    def propagate(self, X, atype='sigmoid'):

        H = self.pre_activation(X)
        H = activation_fn_th(H, atype=atype)
        return H

    def pre_activation(self, X):

        Z = self.post_batch_norm(X, testF=testF)
        H = self.eta * Z + self.beta
        return H

    def post_batch_norm(self, X):

        Z = T.dot(X, self.W) 
        mean    = Z.mean(axis=-1)
        std     = Z.std( axis=-1)
        Z       = (Z - mean) / (std + TINY)

        return Z


def layer_norm_fn(Z, beta, eta):
   
    
    mean, var = tf.nn.moments(Z,axes=[1])
    Z       = (Z - tf.expand_dims(mean, 1)) / \
                   tf.sqrt(tf.expand_dims(var,1) + TINY)
    
    H       = tf.expand_dims(eta, 0) * Z + tf.expand_dims(beta, 0)
    return H

