import math
import numpy as np
import tensorflow as tf

from collections import deque

from base import BaseModel
from model_kits.layer import Layer 
from model_kits.rnn_layer import RNNLayer
from model_kits.lstm_layer import LSTMLayer
from model_kits.bilstm_layer import BiLSTMLayer
from model_kits.embedding_layer import EMBED_Layer

from utils.tf_utils import last_relevant

class CHAR_SEQ2SEQ(BaseModel):

    def __init__(self, model_config, network=None, summary_writer=None):
        super(CHAR_SEQ2SEQ, self).__init__(model_config)

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
        self._define_topology(network)


    def _define_topology(self, network):

        with tf.variable_scope(self.scope_name):

            self.network, self.params = [], []
            if network is not None:
                self.network = network
                for layer in self.network: self.params += layer.params 

            else:

                self.embed_layer = EMBED_Layer(self.D, self.num_hids[1], 'embedding_layer')
                #TODO : 
                # At the moment, only single layer encoder and decoder
                # But should be flexible enough to have deeper layers
                # when they are required
                self.encoder     = [LSTMLayer(  self.num_hids[1], \
                                                self.num_hids[2], \
                                                'char_encoder_l1') , \
                                    LSTMLayer(  self.num_hids[2],
                                                self.num_hids[3],
                                                'char_encoder_l2')] 

                self.decoder     = [LSTMLayer(  self.num_hids[1], \
                                                self.num_hids[2], \
                                                'char_decoder_l1') , \
                                    LSTMLayer(  self.num_hids[2],
                                                self.num_hids[3],
                                                'char_decoder_l2')]

                self.softmax_layer = Layer(self.num_hids[3], self.num_hids[-1],\
                                     'softmax', 'l'+str(self.num_layers))


                self.params  = self.embed_layer.params  + self.encoder[0].params +\
                               self.decoder[0].params   + self.softmax_layer.params 

                self.network = [self.embed_layer] + \
                                self.encoder + \
                                self.decoder + \
                               [self.softmax_layer]
    

    def __call__(self, X):

        with tf.variable_scope(self.scope):

            #TODO: consider outputting error 
            return self.fp(X)

    def cost(self, X, Y, Hts_init=None, Ots_init=None, batch_sz=None, lam=0.0005):
      
        if batch_sz is None: batch_sz = self.batch_sz 
        if Hts_init is None: Hts_init, Ots_init = self._init_states(Ots_init)
        if num_steps is None: num_steps = self.TT 

        preds, logits, _, _ = self.fp(X, Hts_init, Ots_init)

        ## Measured based on perplexity - measures how surprised the network
        ## is to see the next character in a sequence.
        loss        = tf.nn.seq2seq.sequence_loss_by_example(logits,
                       [tf.reshape(Y, [-1])],
                       [tf.ones([batch_sz * self.TT])],
                       self.vocab_sz)
    
        cost = tf.reduce_sum(loss) / batch_sz / self.TT
        l2_loss = tf.add_n([tf.nn.l2_loss(v) \
                        for v in tf.trainable_variables()])

        with tf.variable_scope('summary'):
            tf.histogram_summary("prediction_error", preds)
            tf.scalar_summary("Cost", cost)
            self.summarize = tf.merge_all_summaries() 

        return cost + lam * l2_loss

    
    def cost(self, X, Y, XXM, YYM, batch_sz=None, num_steps=None, lam=0.0005):
        ''' Returns loss
                X - source indice
                Y - target indice
                
                Note that number of batch size is not fixed per update'''

        
        if batch_sz is None : batch_sz = tf.shape(Y)[0]
        if num_steps is None: num_steps = self.TT 

        preds, logits, _ = self.fp(X, XXM, batch_sz, \
                    num_fsteps=num_steps, num_bsteps=num_steps)

        ## Measured based on perplexity - measures how surprised the network
        ## is to see the next character in a sequence.
        loss    = tf.nn.seq2seq.sequence_loss_by_example(logits,
                       [tf.reshape(Y, [-1])],
                       [tf.ones([batch_sz * num_steps])],
                       self.vocab_sz)
  
        Y_len   = tf.cast(tf.reduce_sum(YYM, 1), 'float32')
        cost    = tf.reduce_sum(loss * YYM, 1) / Y_len / tf.to_float(batch_sz)
        #l2_loss = tf.add_n([tf.nn.l2_loss(v) \
        #                for v in tf.trainable_variables()])

        with tf.variable_scope('summary'):
            tf.histogram_summary("prediction_error", preds)
            tf.scalar_summary("Cost", cost)
            self.summarize = tf.merge_all_summaries() 

        return cost #+ lam * l2_loss

   
    def encoder_fp(self, X, Hts, Ots=None, num_steps=None):

        if num_steps is None: num_steps = self.TT 

        Yts, Cts = [], []
        with tf.variable_scope(self.scope_name):

            for t in xrange(num_steps):
                
                Xt = X[:,t]
                Et = self.embed_layer.fp(Xt)
                Hts, Ots = self.step(Et, Hts, Ots, t)

            return Hts, Ots
   

    def decoder_fp(self, Ct, Hts, Ots=None, num_steps=None):
        """ Decoder RNN 
            S           - Summary vector
            Hts, Ots    - initial state of the RNN
            num_steps   - number of total steps """ 

        if num_steps is None: num_steps = self.TT 

        with tf.variable_scope(self.scope_name):

            for t in xrange(num_steps):

                Et = self.embed_layer.fp(Ct)
                Hts, Ots = self.step(Et, Hts, Ots, t)
                Ct = self.get_character(Hts[-1][-1], stype='argmax')

            YYY     = tf.reshape(tf.concat(1, Hts[-1]), [-1, self.num_hids[-2]])
            logits  = self.softmax_layer.get_logit(YYY)
            preds   = tf.nn.softmax(logits)

            return preds, [logits], Hts


    def step(self, Et, Hts, Ots, tt):

        Ot = Et
        for i in xrange(len(self.encoder)):
        
            Ht = Hts[i][-1]
            Ot, Ht = self.encoder[i].fp(Ot, Ht)

            Ots[i].append(Ot)
            Hts[i].append(Ht)

        return Hts, Ots


    def fp(self, X, batch_sz, num_fsteps, num_bsteps):

        #Encoder pass
        Hts_enc, Ots_enc = self._init_states(batch_sz, num_fsteps)
        Hts_enc, Ots_enc = self.encoder_fp(X, Hts_enc, \
                                    Ots_enc, num_steps=num_fsteps)

        #Decoder hidden initialization
        #TODO : Unsure about whether i should subtract by -1 or -2
        #       to compute the last index.
        Hts_dec, Ots_dec = [],[]
        last_indice = tf.cast(tf.reduce_sum(XXM, 1)-1, 'int32')

        for i in xrange(len(Hts_enc)):

            Hts = tf.pack(Hts_enc[i])
            Ots = tf.pack(Ots_enc[i])
            Hts_init = last_relevant(Hts, last_indice)
            Ots_init = last_relevant(Ots, last_indice)

            Hts_dec.append([Hts_init])
            Ots_dec.append([Ots_init])

        #Decoder pass
        C0 = self.get_character(Hts_init, stype='argmax')
        pred, logits, Zts = self.decoder_fp(C0, Hts_dec, \
                                    Ots=Ots_dec, num_steps=num_bsteps)

        return pred, logits, Zts

    def get_character(self, Ht, stype='argmax'):
        """stype - sample type, either 'argmax' | 'multinomial' """


        #Get prediction
        pred    = self.softmax_layer.fp(Ht)
       
        #Sample a character
        if stype == 'multinomial':
            pred = tf.multinomial(pred, 1, seed=1234, name=None)
        
        sample_char = tf.argmax(pred, 1)
        symbol  = tf.stop_gradient(sample_char)
 
        return symbol


    def get_fp_embedding(self, X, _):

        pred    = self.softmax_layer.fp(X)
        symbol  = tf.stop_gradient(tf.argmax(pred, 1))

        return self.embed_layer.fp(symbol)


    def get_context(self, X, Hts_init=None, Ots_init=None, num_steps=None):
        """
        Returns the last hidden representation H_T (feature) of the RNN 
        with input X = {x_1, ..., x_T}

        Note that lstm layer returns two item, ouputs Ots and hiddens Hts,
        where as RNN returns one item, hiddens Hts (outputs and hiddens are the same).
        Hence, argument Ots is an option when lstm layer is used.
        """

        if num_steps is None: num_steps = self.TT 
        if Hts_init is None: Hts_init, Ots_init = self._init_states('bilstm')

        Ht, Hts, Ots  = self.encoder_fp(X, Hts_init, Ots=Ots_init, num_steps=num_steps)
        C0 = self.get_character(Ht, stype='argmax')
        return C0, Hts, Ots 


    def sample(self, C0, num_steps, Hts=None, Ots=None, stype='argmax'):
        """Generates text
            S           - summary variable
            num_steps   - number of total character generated 
            Hts, Ots    - is dummy variable. Needed this to synchronize 
            with sample method in char_rnn.py """

        #TODO : At test time, the prediction is currently done iteratively 
        #character by character in a greedy fashion, but eventually needs to be
        #implemented more sophisticated methods (e.g. beam search).
        _, _, Yts = self.decoder_fp(C0, Hts, Ots=Ots, num_steps=num_steps)

        char_inds = []
        for yt in Yts:
            ct = self.get_character(yt, stype=stype)
            char_inds.append(ct)

        return char_inds


    def _init_states(self, batch_sz, num_steps):

        Hts, Ots = [], []
        for i in xrange(1, self.num_layers-1):
            Hts.append([tf.zeros([batch_sz, self.num_hids[i]])])
            Ots.append([tf.zeros([batch_sz, self.num_hids[i]])])

        return Hts, Ots
                
                
    def clone(self, new_scope_name=None):

        #TODO
        pass



