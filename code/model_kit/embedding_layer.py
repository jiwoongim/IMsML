import math
import tensorflow as tf

from utils.utils import base_name
from utils.nn_utils import init_weights, activation_fn


class EMBED_Layer(object):

    def __init__(self, D, M, scope_name):

        self.D = D   # Vocabulary size
        self.M = M   # Embedding Output size
        self.scope_name = scope_name

        self._initialize_params()

    def _initialize_params(self):
        '''Initialize parameters in the layer'''

        with tf.variable_scope(self.scope_name) as sc:

            self.W_embed = tf.get_variable("W_embed", shape=[self.D, self.M], \
                    initializer=init_weights('xavier'))
            self.params = [self.W_embed]

    def __call__(self, X):

        with tf.variable_scope(self.scope):     
            
            return self.fp(X)


    def fp(self, X):

        with tf.variable_scope(self.scope_name):

            return tf.nn.embedding_lookup(self.W_embed, X)


