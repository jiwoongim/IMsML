import os,sys
import numpy as np
import tensorflow as tf 

from model_kits.layer_norm import layer_norm_fn
from utils.nn_utils import *
from utils.tf_utils import *


class GRU_LN_Layer():

    def __init__(self, D, M, layer):
        self.layer=layer
        self.M = M
        self.D = D
        self.initialize_recurrent_params(D,M)

    # LN-GRU layer
    def initialize_recurrent_params(self, D, M ):
        """
        Gated Recurrent Unit (GRU) with LN
        """
        self.W = tf.get_variable('W_ln'+self.layer, \
                                shape=[self.D, self.M * 2], \
                                initializer=init_weights('xavier'))
        self.U = tf.get_variable('U_ln'+self.layer, \
                                shape=[self.M, self.M * 2], \
                                initializer=init_weights('xavier'))
        self.x = tf.Variable(tf.zeros([self.M * 2,], tf.float32), 'xbias'+self.layer)
 

        self.Wx = tf.get_variable('Wx_ln'+self.layer, \
                                  shape=[self.D, self.M], \
                                  initializer=init_weights('xavier'))
        self.Ux = tf.get_variable('Ux_ln'+self.layer, \
                                  shape=[self.M, self.M], \
                                  initializer=init_weights('xavier'))
        self.bx = tf.Variable(tf.zeros([self.M,], tf.float32), 'hbias'+self.layer)
        #TODO U and Ux needs be concatenation of two orthogoanl matrix, not xavier



        # LN parameters
        scale_add = 0.0
        scale_mul = 1.0
        self.b1 = tf.Variable(scale_add * tf.ones([2*M,], tf.float32), 'b1_ln_'+self.layer)
        self.b2 = tf.Variable(scale_add * tf.ones([1*M,], tf.float32), 'b2_ln_'+self.layer)
        self.b3 = tf.Variable(scale_add * tf.ones([2*M,], tf.float32), 'b3_ln_'+self.layer)
        self.b4 = tf.Variable(scale_add * tf.ones([1*M,], tf.float32), 'b4_ln_'+self.layer)
        self.s1 = tf.Variable(scale_mul * tf.ones([2*M,], tf.float32), 's1_ln_'+self.layer)
        self.s2 = tf.Variable(scale_mul * tf.ones([1*M,], tf.float32), 's2_ln_'+self.layer)
        self.s3 = tf.Variable(scale_mul * tf.ones([2*M,], tf.float32), 's3_ln_'+self.layer)
        self.s4 = tf.Variable(scale_mul * tf.ones([1*M,], tf.float32), 's4_ln_'+self.layer)

	self.params = [ self.W,  self.U,  self.x, \
		 	self.Wx, self.Ux, self.bx, \
			self.b1, self.b2, self.b3,self.b4,\
			self.s1, self.s2, self.s3,self.s4 ] 


    def step(self, h_tm1, XX):

        x_  = XX[:,:self.M*2]
        xx_ = XX[:,self.M*2:] 
    	def _slice(_x, n, dim):
    	    if tf.shape(_x) == 3:
    	        return _x[:, :, n*dim:(n+1)*dim]
    	    return _x[:, n*dim:(n+1)*dim]

        x_  = layer_norm_fn(x_ , self.b1, self.s1)
        xx_ = layer_norm_fn(xx_, self.b2, self.s2)

        preact = tf.matmul(h_tm1, self.U)
        preact = layer_norm_fn(preact, self.b3, self.s3)
        preact = preact + x_

        r = tf.nn.sigmoid(_slice(preact, 0, self.M))
        u = tf.nn.sigmoid(_slice(preact, 1, self.M))

        preactx = tf.matmul(h_tm1, self.Ux)
        preactx = layer_norm_fn(preactx, self.b4, self.s4)
        preactx = preactx * r + xx_

        h = tf.tanh(preactx)
        h = u * h_tm1 + (1. - u) * h

        return h


    def propagate(self, X, n_steps=1, h0=None, scanF=True):
        """
        Feedforward pass through GRU with LN
        """
        n_samples = X.get_shape()[1]
 
        if h0 is None: h0 = tf.alloc(0., n_samples, self.M)

        x_ = tf.matmul(tf.reshape(X, [-1, self.D]), self.W)  + self.x
        xx_= tf.matmul(tf.reshape(X, [-1, self.D]), self.Wx) + self.bx

        ## It is important that this block of code is after x_, xx_ assigment
        if n_steps == 1:
            XX = tf.concat(1, [x_,xx_])
            return self.step(h0, XX)
 

        shape0 = X.get_shape()[:2]
        shape1 = X.get_shape()[:2]

        x_ = tf.reshape(x_ , tf.concat(0, [shape0, [self.M*2]]))
        xx_= tf.reshape(xx_, tf.concat(0, [shape1, [self.M]  ]))
        XX = tf.concat(2, [x_,xx_])
   
        if scanF:
            rval = tf.scan(self.step, XX, initializer=h0)

        else:
            rval  = []
            h_tm1 = h0
            for i in xrange(int(n_steps)):

                h_tm1 = self.step(h_tm1, x_, xx_)
                rval.append(h_tm1)

        return rval



