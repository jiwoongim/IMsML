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


    def encoder(self, X, batch_sz, num_steps=None):

        if num_steps is None: num_steps = self.TT 

        #Encoder pass
        E       = self.embed_layer.fp(X)
        Hts_enc = self._init_states(self.encoder, batch_sz)
        Hts_enc = self.estep(E, Hts_enc, num_steps)
        return Hts_enc


    def estep(self, E, Hs, num_esteps=None):

        if num_esteps is None: num_esteps = self.TT 

        Oi = tf.transpose(E, perm=[1,0,2])
        with tf.variable_scope(self.scope_name):
            for i in xrange(len(self.encoder)):
            
                Hi = Hs[i]
                Oi = self.encoder[i].propagate(Oi, h0=Hi,\
                                     n_steps=tf.cast(num_esteps, 'int32'))
            return Oi


    def decoder(self, predt, Hts, Ots=None, num_steps=None, scanF=False):
        """ Decoder RNN 
            S           - Summary vector
            Hts, Ots    - initial state of the RNN
            num_steps   - number of total steps """ 

        if num_steps is None: num_steps = self.TT 

        with tf.variable_scope(self.scope_name):

            if scanF:
                O10, O20 = Hts
                def decode_t(ctm1, h10, h20):

                    Et = self.embed_layer.fp(ctm1)
                    O1t = self.decoder[0].propagate(Et , n_steps=1, h0=h10)
                    O2t = self.decoder[1].propagate(O1t, n_steps=1, h0=h20)
                    predt = self.pred_layer.propagate(O2t, atype='softmax')
                    return predt, O1t, O2t

                [preds, Hts1, Hts2], updates=tf.scan(decode_t,
                                          outputs_info = [C0, O10, O20],
                                          n_steps=tf.cast(num_steps, dtype='int32'))
            else:
                O1t, O2t = Hts
                preds = []
                for t in xrange(num_steps):

                    Et = self.embed_layer.fp(predt)
                    O1t = self.decoder[0].propagate(Et , n_steps=1, h0=O1t)
                    O2t = self.decoder[1].propagate(O1t, n_steps=1, h0=O2t)
                    pred = self.softmax_layer.propagate(O2t, atype='softmax')
                    preds.append(predt)
                preds = tf.pack(preds)
            return preds


    def fp(self, X, batch_sz, num_esteps, num_dsteps):

        if num_esteps is None: num_esteps = self.TT 
        if num_dsteps is None: num_dsteps = self.TT 

        #Encoder pass
        Hts_dec, Ots_dec = [],[]
        Hts_dec_init = self.encoder(X, batch_sz, num_steps=num_esteps)
        Hts_dec      = self._init_states(self.decoder, batch_sz)
        Hts_dec.pop(0)
        Hts_dec.insert(0, Hts_dec_init)

        #Decoder pass
        #C0 = last_relevant2D(X, last_indice)
        pred0 = tf.constant(np.ones((tr_config['batch_sz'],), dtype='int32'))
        pred = self.decoder(pred0, Hts_dec, num_steps=num_dsteps)

        return pred

