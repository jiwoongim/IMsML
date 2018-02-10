import math
import numpy as np
import tensorflow as tf

from base import BaseModel
from model_kits.layer import Layer

class MLP(BaseModel):
    
    def __init__(self, model_params, config, network=None):
        super(MLP, self).__init__(config)

        self.config = config
        self.D, self.num_hids, self.O, self.scope_name, self.atypes =\
                model_params['num_hids'][0] , model_params['num_hids']  ,\
                model_params['num_hids'][-1], model_params['scope_name'],\
                model_params['atypes']
       
        #Defining Model Topology
        with tf.variable_scope(self.scope_name):

            self.network, self.params = [], []
            if network is not None:
                self.network = network
                for layer in self.network: self.params += layer.params 

            else:
                for i in xrange(len(self.num_hids)-1):
                    self.network.append(Layer(\
                            self.num_hids[i], self.num_hids[i+1],\
                            self.atypes[i]  , 'l'+str(i)))
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

        return MLP(model_params, self.config, network=network_clone)



