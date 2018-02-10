import math
import numpy as np
import tensorflow as tf

from base import BaseModel
from model_kits.layer import Layer, Conv_Layer


class Convnet(BaseModel):

    def __init__(self, model_params, config, network=None):
        super(Convnet, self).__init__(config)

        self.config = config
        self.Xdim, self.num_hids, self.O, self.strides, \
                                  self.scope_name, self.atypes =\
                model_params['num_hids'][0] , model_params['num_hids']  ,\
                model_params['num_hids'][-1], model_params['num_stides'], \
                model_params['scope_name']  , model_params['atypes']

        #Defining Model Topology
        with tf.variable_scope(self.scope_name):

            self.network, self.params = [], []
            if network is not None:
                self.network = network
                for layer in self.network: self.params += layer.params 

            else:

                layer_dim = self.Xdim 
                for i in xrange(1, len(self.num_hids)-1):

                    # Fully connected layer
                    if len(self.num_hid[i]) == 1:   

                        # Case when previous layer was convolutional layer
                        layer_dim = num_hids[-1]
                        if len(layer_dim) != 3:     

                            layer_dim = np.prod(self.num_hid[-1]) / \
                                        np.prod (self.strides[-1])
                            
                            
                        layer_i = Layer(layer_dim           , \
                                        self.num_hids[i]  ,\
                                        self.atypes[i]      , \
                                        'l'+str(i))

                    # Convolutional layer:
                    # 3 Dim. corresponds to (num_kern, filter_sz_x, filter_sz_y)
                    elif len(self.num_hid[i]) == 3: 

                         #TODO layer_dim
                         layer_i = Conv_Layer(  layer_dim,
                                                self.num_hids[i][0], \
                                                self.num_hids[i][1:],\
                                                self.atypes[i], \
                                                self.strides[i],\
                                                'conv_l'+str(i)  )

                        
                    self.network.append(layer_i)
                    self.params += self.network[i].params


    def __call__(self, X):

        with tf.variable_scope(self.scope):

            #TODO: consider outputting error 
            return self.fp(X)


    def fp(self, X):

        H_i = X
        with tf.variable_scope(self.scope_name):       

            for i in xrange(len(self.network)): 
                H_i = self.network[i].fp(H_i, self.atypes[i])
        
        return H_i



    def clone(self, new_scope_name=None):

        if new_scope_name is None: new_scope_name = self.scope_name + "_clone"

        network_clone = [layer.clone() for layer in self.network]
        model_params  = {'num_hids':self.num_hids, \
                        'atypes':self.atypes, 'scope_name':new_scope_name}

        return Convnet(model_params, self.config, network=network_clone)



