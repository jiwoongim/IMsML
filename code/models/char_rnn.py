import math
import numpy as np
import tensorflow as tf

from collections import deque
#from tf.nn.rnn_cell import rnn_cell
#from tensorflow.models.rnn import seq2seq

from models.base import BaseModel
from models.mlp import MLP 
from model_kits.layer import Layer 
from model_kits.rnn_layer import RNNLayer 
from model_kits.lstm_layer import LSTMLayer 
from model_kits.embedding_layer import EMBED_Layer

class CHAR_RNN(BaseModel):

    def __init__(self, model_config, network=None, summary_writer=None):
        super(CHAR_RNN, self).__init__(model_config)

        self.model_config   = model_config
        self.vocab_sz       = model_config['vocab_sz']
        self.num_hids       = model_config['num_hids']
        self.D              = model_config['num_hids'][0] 
        self.O              = model_config['num_hids'][-1]
        self.TT             = model_config['seq_length']
        self.scope_name     = model_config['scope_name'] or 'char_rnn'
        self.rnn_type       = model_config['rnn_type']
        self.atype          = model_config['atype']
        self.batch_sz       = model_config['batch_sz']
        self.num_layers     = len(self.num_hids)-1
        self.summary_writer = summary_writer
        self.ind2voc        = model_config['ind2voc']
        self.voc2ind        = model_config['voc2ind']


        #Defining Model Topology
        with tf.variable_scope(self.scope_name):

            self.network, self.params = [], []
            if network is not None:
                self.network = network
                for layer in self.network: self.params += layer.params 

            else:
    
                self.embed_layer = EMBED_Layer(self.D, self.num_hids[1], 'embedding_layer')
                self.network.append(self.embed_layer)

                for i in xrange(1,self.num_layers-1):
                    if self.rnn_type == 'rnn' or self.rnn_type == 'irnn':

                        ## TODO 
                        # The implementation of RNN has not tested at all. This is 
                        # not urgent since we don't use RNN (no one uses RNN these days)
                        self.rnn_layer_i = RNNLayer(self.num_hids[i], self.num_hids[i+1], self.atype, 'rnn_ly'+str(i))
            
                    elif self.rnn_type == 'lstm':
                        self.rnn_layer_i = LSTMLayer(self.num_hids[i], self.num_hids[i+1], 'rnn_ly'+str(i))

                    ## TODO GRU not available yet

                    self.network.append(self.rnn_layer_i)
                    self.params += self.rnn_layer_i.params

                self.softmax_layer = Layer(self.num_hids[-2], self.num_hids[-1],\
                                     'softmax', 'l'+str(self.num_layers))

                self.params  =  self.embed_layer.params +\
                                self.softmax_layer.params 

                self.network.append(self.softmax_layer)


    def __call__(self, X, Hts, Ots=None):

        with tf.variable_scope(self.scope):

            #TODO: consider outputting error 
            return self.fp(X, Hts, Ots)


    def fp(self, X, Hts, Ots=None, num_steps=None):
        """ Forward pass """

        if num_steps is None: num_steps = self.TT 
        with tf.variable_scope(self.scope_name):

            ## Embedding Layer
            #XX  = tf.split(1, num_steps, self.embed_layer.fp(X))
            #XXl = [tf.squeeze(input_, [1]) for input_ in XX]
           
            ## Intermediate Layers
            Yts = []
            for t in xrange(num_steps):
                Xt = X[:,t]
                Et = self.embed_layer.fp(Xt)
                Ot, Hts, Ots = self.step(Et, Hts, Ots)
                Yts.append(Ot)

            ## Softmax Layer 
            YYY     = tf.reshape(tf.concat(1, Yts), [-1, self.num_hids[-2]])
            logits  = self.softmax_layer.get_logit(YYY)
            preds   = tf.nn.softmax(logits)
            
            return preds, [logits], Hts, Ots


    def cost(self, X, Y, Hts_init=None, Ots_init=None, batch_sz=None, lam=0.0005, TT=None):
      
        if batch_sz is None: batch_sz = self.batch_sz 
        if Hts_init is None: Hts_init, Ots_init = self._init_states(Ots_init)

        preds, logits, _, _ = self.fp(X, Hts_init, Ots_init, num_steps=TT)

        ## Measured based on perplexity - measures how surprised the network
        ## is to see the next character in a sequence.
        loss        = tf.nn.seq2seq.sequence_loss_by_example(logits,
                       [tf.reshape(Y, [-1])],
                       [tf.ones([batch_sz * self.TT])],
                       self.vocab_sz)
   
        cost    = tf.reduce_sum(loss) / tf.to_float(batch_sz) / tf.to_float(self.TT)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) \
                        for v in tf.trainable_variables()])

        with tf.variable_scope('summary'):
            tf.histogram_summary("prediction_error", preds)
            tf.scalar_summary("Cost", cost)
            self.summarize = tf.merge_all_summaries() 

        return cost + lam * l2_loss

    
    def step(self, Xt, Hts, Ots=None):

        Ot = Xt 
        for i in xrange(1, self.num_layers-1):

            Ht = Hts.popleft()
            if self.rnn_type == 'lstm':
                    
                Ot, Ht = self.network[i].fp(Ot, Ht)
                #if i != self.num_layers-2:
                #    Ots.append(Ot)
                #    Ot = Ots.popleft() # Pop init_y from lstm
            else:
                Ht = self.network[i].fp(Xt, Ht)
            Hts.append(Ht)
    
        return Ot, Hts, Ots


    def get_context(self, X, Hts_init=None, Ots_init=None, num_steps=None):
        """
        Returns the last hidden representation H_T (feature) of the RNN 
        with input X = {x_1, ..., x_T}

        Note that lstm layer returns two item, ouputs Ots and hiddens Hts,
        where as RNN returns one item, hiddens Hts (outputs and hiddens are the same).
        Hence, argument Ots is an option when lstm layer is used.
        """

        if num_steps is None: num_steps = self.TT 
        if Hts_init is None: Hts_init, Ots_init = self._init_states(Ots_init)

        preds, logits, Hts, Ots  = self.fp(X,   Hts=Hts_init, \
                                                Ots=Ots_init, \
                                                num_steps=num_steps)
        symbols                  = tf.stop_gradient(tf.argmax(preds, 1))
        
        return tf.reshape(symbols[num_steps-1], [1]), Hts, Ots 


    def sample(self, Ct, num_steps, Hts=None, Ots=None):
        """
        Generates sample of text

        Note that lstm layer returns two item, ouputs Ots and hiddens Hts,
        where as RNN returns one item, hiddens Hts (outputs and hiddens are the same).
        Hence, argument Ots is an option when lstm layer is used.
        """

        char_inds = [Ct]

        #TODO : At test time, the prediction is currently done iteratively 
        #character by character in a greedy fashion, but eventually needs to be
        #implemented more sophisticated methods (e.g. beam search).
        with tf.variable_scope(self.scope_name):

            for t in xrange(num_steps):
                Xt              = self.embed_layer.fp(Ct)
                Ht, Hts, Ots    = self.step(Xt, Hts, Ots)
                Ht              = tf.reshape([Ht], [-1, self.num_hids[-2]])
                Ct              = self.get_character(Ht, stype='argmax')
                char_inds.append(Ct)

            return char_inds


    def get_character(self, Ht, stype='multinomial'):
        """stype - sample type, either 'argmax' | 'multinomial' """


        #Get prediction
        pred    = self.softmax_layer.fp(Ht)
       
        #Sample a character
        if stype == 'multinomial':
            pred = tf.multinomial(pred, 1, seed=1234, name=None)
        
        symbol  = tf.stop_gradient(tf.argmax(pred, 1)) 
        return symbol

   
    def clone(self, new_scope_name=None):
        
        #TODO
        pass


    def _init_states(self, Ots_init=None):
        Hts_init = deque()
        for i in xrange(2, self.num_layers): 
            Hts_init.append(tf.zeros([1, self.num_hids[i]], tf.float32))

        if self.rnn_type == 'lstm':
            Ots_init = deque()
            for i in xrange(2, self.num_layers-1): 
                Ots_init.append(tf.zeros([1, self.num_hids[i]], tf.float32))

        return Hts_init, Ots_init

