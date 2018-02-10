import math
import numpy as np
import tensorflow as tf

from collections import deque

from aif_ml_lib.tensorflow.models.base import BaseModel
from aif_ml_lib.tensorflow.model_kits.layer import Layer 
from aif_ml_lib.tensorflow.model_kits.rnn_layer import RNNLayer
from aif_ml_lib.tensorflow.model_kits.lstm_layer import LSTMLayer
from aif_ml_lib.tensorflow.model_kits.bilstm_layer import BiLSTMLayer
from aif_ml_lib.tensorflow.model_kits.embedding_layer import EMBED_Layer

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
                self.encoder     = [BiLSTMLayer(self.num_hids[1], \
                                                self.num_hids[2], \
                                                'char_encoder')]
                self.decoder     = [LSTMLayer(  self.num_hids[2], \
                                                self.num_hids[3], \
                                                'char_decoder')]
                self.pca_layer_H = Layer(self.num_hids[2] * 2, self.num_hids[2],\
                                                    'linear', 'pca_H')

                self.softmax_layer = Layer(self.num_hids[3], self.num_hids[-1],\
                                     'softmax', 'l'+str(self.num_layers))


                self.params  = self.embed_layer.params  + self.encoder[0].params +\
                               self.decoder[0].params   + self.softmax_layer.params +\
                               self.pca_layer_H.params  

                self.network = [self.embed_layer] + \
                                self.encoder + \
                                self.decoder + \
                               [self.pca_layer_H, \
                                self.softmax_layer]
    

    def __call__(self, X):

        with tf.variable_scope(self.scope):

            #TODO: consider outputting error 
            return self.fp(X)

   
    def encoder_fp(self, X, Hts, num_steps=None):

        if num_steps is None: num_steps = self.TT 

        with tf.variable_scope(self.scope_name):

            #XX  = tf.split(1, num_steps, self.embed_layer.fp(X))
            #XXl = [tf.squeeze(input_, [1]) for input_ in XX]

            for t in xrange(num_steps):

                Xft, Xbt = X[:,t], X[:,num_steps-t-1]
                Eft, Ebt = self.embed_layer.fp(Xft), self.embed_layer.fp(Xbt) 
                Ot, Ht, Hts = self.fstep([Eft, Ebt], Hts)

            #tf.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name=None)

            HT_enc = self.pca_layer_H.fp(tf.concat(1, [Ht[0], Ht[1]]))

            return HT_enc
   

    def fstep(self, Et, Hts):

        Oft, Obt = Et[0], Et[1]

        for i in xrange(len(self.encoder)):

            Hft, Hbt = Hts[0].popleft(), Hts[1].popleft()
            Hft, Oft = self.encoder[i].flstm.fp(Oft, Hft)
            Hbt, Obt = self.encoder[i].blstm.fp(Obt, Hbt)
            Hts[0].append(Hft), Hts[1].append(Hbt)

        return [Oft,Obt], [Hft, Hbt], Hts


    def decoder_fp(self, S, Hts, num_steps=None):
        """ Decoder RNN 
            S           - Summary vector
            Hts, Ots    - initial state of the RNN
            num_steps   - number of total steps """ 

        Yts = []
        if num_steps is None: num_steps = self.TT 

        with tf.variable_scope(self.scope_name):

            for t in xrange(num_steps):
                Ht, _ = self.bstep(S, Hts)
                Yts.append(Ht)

            YYY     = tf.reshape(tf.concat(1, Yts), [-1, self.num_hids[-2]])
            logits  = self.softmax_layer.get_logit(YYY)
            preds   = tf.nn.softmax(logits)

            return preds, [logits], Yts


    def bstep(self, Xt, Hts):

        Ot=Xt
        for i in xrange(len(self.decoder)):

            # Pop init_y from lstm
            Ht = Hts.popleft()
            Ot, Ht = self.decoder[i].fp(Ot, Ht)
            Hts.append(Ht)

        return Ht, Hts


    def fp(self, X, num_fsteps, num_bsteps=None):

        if num_bsteps is None: num_bsteps = num_fsteps 

        #Encoder pass
        Hts_init_enc = self._init_states('bilstm')
        HT_enc = self.encoder_fp(X, Hts_init_enc, num_steps=num_fsteps)


        #Decoder pass
        Hts_init_dec = self._init_states('lstm')
        pred, logits, Zts = self.decoder_fp(HT_enc, Hts_init_dec, \
                                num_steps=num_bsteps)

        return pred, logits, Zts

    
    def cost(self, X, Y, batch_sz=None, num_steps=None, lam=0.0005):

        if batch_sz is None: batch_sz = self.batch_sz 
        if num_steps is None: num_steps = self.TT 
        preds, logits, _ = self.fp(X, self.TT)

        ## Measured based on perplexity - measures how surprised the network
        ## is to see the next character in a sequence.
        loss        = tf.nn.seq2seq.sequence_loss_by_example(logits,
                       [tf.reshape(Y, [-1])],
                       [tf.ones([batch_sz * num_steps])],
                       self.vocab_sz)
        
        cost = tf.exp(tf.reduce_sum(loss) / tf.to_float(batch_sz) / tf.to_float(num_steps))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) \
                        for v in tf.trainable_variables()])

        with tf.variable_scope('summary'):
            tf.histogram_summary("prediction_error", preds)
            tf.scalar_summary("Cost", cost)
            self.summarize = tf.merge_all_summaries() 

        return cost + lam * l2_loss


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


    def get_context(self, X, Hts_init=None, num_steps=None):
        """
        Returns the last hidden representation H_T (feature) of the RNN 
        with input X = {x_1, ..., x_T}

        Note that lstm layer returns two item, ouputs Ots and hiddens Hts,
        where as RNN returns one item, hiddens Hts (outputs and hiddens are the same).
        Hence, argument Ots is an option when lstm layer is used.
        """

        if num_steps is None: num_steps = self.TT 
        if Hts_init is None: Hts_init = self._init_states('bilstm')

        S, Hts  = self.encoder_fp(X, Hts_init, num_steps=num_steps)
        return S, Hts 


    def sample(self, S, num_steps, Hts=None, stype='argmax'):
        """Generates text
            S           - summary variable
            num_steps   - number of total character generated 
            Hts, Ots    - is dummy variable. Needed this to synchronize 
            with sample method in char_rnn.py """

        #TODO : At test time, the prediction is currently done iteratively 
        #character by character in a greedy fashion, but eventually needs to be
        #implemented more sophisticated methods (e.g. beam search).
        Hts = self._init_states('lstm')
        _, _, Yts = self.decoder_fp(S, Hts, num_steps=num_steps)

        char_inds = []
        for yt in Yts:
            ct = self.get_character(yt, stype=stype)
            char_inds.append(ct)

        return char_inds


    def _init_states(self, ltype):

        if ltype == 'bilstm':
            Hts_init = [deque(), deque()]
            for i in xrange(2,len(self.encoder)+2): 
                Hts_init[0].append(tf.zeros([1, self.num_hids[2]], tf.float32))
                Hts_init[1].append(tf.zeros([1, self.num_hids[2]], tf.float32))

            return Hts_init

        elif ltype == 'lstm':
            Hts_init = deque()
            till = len(self.encoder)+len(self.decoder)+2
            for i in xrange(len(self.encoder)+2, till): 
                Hts_init.append(tf.zeros([1, self.num_hids[i]], tf.float32))

            return Hts_init



    def clone(self, new_scope_name=None):

        #TODO
        pass



