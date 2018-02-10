import math
import numpy as np
import tensorflow as tf

from collections import deque

from base import BaseModel
from model_kits.layer import Layer 
from model_kits.rnn_layer import RNNLayer
from model_kits.ln_gru_layer import GRU_LN_Layer
from model_kits.ln_bigru_layer import BiGRU_LN_Layer
from model_kits.embedding_layer import EMBED_Layer

from utils.tf_utils import last_relevant, extract_last_relevant

class CHAR_SEQ2SEQ_CHAR2WORD_SKIP(BaseModel):

    def __init__(self, model_config, network=None, summary_writer=None):

        self.model_config   = model_config
        self.vocab_sz       = model_config['vocab_sz']
        self.num_hids       = model_config['num_hids']
        self.D              = model_config['num_hids'][0] 
        self.O              = model_config['num_hids'][-1]
        self.TT             = model_config['max_steps']
        self.scope_name     = model_config['scope_name'] or 'char_rnn'
        self.rnn_type       = model_config['rnn_type']
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
                self.encoder     = [BiGRU_LN_Layer(self.num_hids[1], \
                                                   self.num_hids[2],   \
                                                   'char_encoder_l1'), \
                                    BiGRU_LN_Layer(self.num_hids[2], \
                                                   self.num_hids[2],   \
                                                   'char_encoder_l2')] 

                self.decoder     = [GRU_LN_Layer(self.num_hids[2], \
                                                 self.num_hids[2], \
                                                 'char_decoder_l1'), \
                                    GRU_LN_Layer(self.num_hids[2],                                                
                                                 self.num_hids[1],                                                  
                                                 'char_decoder_l2')]

                self.skip_decoder = [GRU_LN_Layer(self.num_hids[2], \
                                                  self.num_hids[2], \
                                                  'char_skip_decoder_l1'), \
                                     GRU_LN_Layer(self.num_hids[2],\
                                                  self.num_hids[1],
                                                  'char_skip_decoder_l2')]


                self.softmax_layer = Layer(self.num_hids[1], self.D,\
                                     'softmax', 'l'+str(self.num_layers))
            
                self.skip_softmax_layer = Layer(self.num_hids[1], \
                                                self.num_hids[-1],\
                                                'softmax',
                                                'skip_l'+str(self.num_layers))


                self.params  = self.embed_layer.params  + \
                               self.encoder[0].params   + self.encoder[1].params +\
                               self.decoder[0].params   + self.decoder[1].params +\
                               self.softmax_layer.params   + \
                               self.skip_decoder[0].params + self.skip_decoder[1].params + \
                               self.skip_softmax_layer.params

                self.network = [self.embed_layer] + \
                                self.encoder + \
                                self.decoder + \
                               [self.softmax_layer] + \
                                self.skip_decoder + \
                               [self.skip_softmax_layer]


    def __call__(self, X):

        with tf.variable_scope(self.scope):

            #TODO: consider outputting error 
            return self.fp(X)


    def load_params(self, np_param_dict):

        assign_ops = []
        for param, np_param_key in zip(self.params, np_param_dict):
            if param.get_shape() == np_param_dict[np_param_key].shape: 
                assign_ops.append(tf.assign(param, np_param_dict[np_param_key]))
            else:
                print 'LOAD FAIL: paramter mis-match'
                print param.name
                print np_param_key
        return  assign_ops
    
    def cost(self, X, Y, XXM, YYM, batch_sz=None, num_steps=None, lam=0.0005):
        ''' Returns loss
                X - source indice
                Y - target indice
                
                Note that number of batch size is not fixed per update'''

        
        if batch_sz is None : batch_sz = tf.shape(Y)[0]
        if num_steps is None: num_steps = self.TT 

        preds = self.fp(X, XXM, batch_sz, \
                    num_esteps=num_steps, num_dsteps=num_steps)
        preds = tf.transpose(preds, perm=[1,0,2])

        ## Measured based on perplexity - measures how surprised the network
        ## is to see the next character in a sequence.

        py      = preds.reshape((batch_sz*num_steps, self.D))
        Y_len   = tf.cast(tf.sum(YYM, 1), 'float32')
        cost    = - tf.log(py)[tf.arange(batch_sz*num_steps), Y.flatten()] * YYM.flatten()
        cost    = cost.reshape((batch_sz, num_steps)) / Y_len.dimshuffle(0,'x')
        cost    = tf.exp(tf.sum(cost, axis=1))
        cost    = tf.sum(cost) / tf.cast(batch_sz, 'float32')

        #l2_loss = tf.add_n([tf.nn.l2_loss(v) \
        #                for v in tf.trainable_variables()])
        with tf.variable_scope('summary'):
            tf.histogram_summary("prediction_error", preds)
            tf.scalar_summary("Cost", cost)
            self.summarize = tf.merge_all_summaries() 

        return cost #+ lam * l2_loss

   
    def encoder_fp(self, X, XXM, batch_sz, num_steps=None):

        if num_steps is None: num_steps = self.TT 

        #Encoder pass
        E       = self.embed_layer.fp(X)
        Hts_enc = self._init_states(self.encoder, batch_sz)
        Hts_enc = self.estep(E, Hts_enc, num_steps)

        #Decoder hidden initialization
        last_indice = tf.cast(tf.reduce_sum(XXM, 1)-1, 'int32')

        #Hts = tf.pack(Hts_enc)
        content = extract_last_relevant(Hts_enc, last_indice+1)

        return content


    def estep(self, E, Hs, num_esteps=None):

        if num_esteps is None: num_esteps = self.TT 

        Oi = tf.transpose(E, perm=[1,0,2])
        with tf.variable_scope(self.scope_name):
            for i in xrange(len(self.encoder)):
            
                Hi = Hs[i]
                Oi = self.encoder[i].propagate(Oi, h0=Hi,\
                                     n_steps=tf.cast(num_esteps, 'int32'))
            return Oi


    def decoder_fp(self, Ct, Hts, decoder, softmax_layer, num_steps=None, scanF=False):
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
                    O1t = decoder[0].propagate(Et , n_steps=1, h0=h10)
                    O2t = decoder[1].propagate(O1t, n_steps=1, h0=h20)
                    pred = softmax_layer.propagate(O2t, atype='softmax')
                    ct   = self.get_character(pred, stype='argmax')
                    return ct, O1t, O2t, pred

                [Cts, Hts1, Hts2, preds], updates=tf.scan(decode_t,
                                          outputs_info = [C0, O10, O20, None],
                                          n_steps=tf.cast(num_steps, dtype='int32'))
            else:
                O1t, O2t = Hts
                preds = []
                for t in xrange(num_steps):

                    Et = self.embed_layer.fp(Ct)
                    O1t = decoder[0].propagate(Et , n_steps=1, h0=O1t)
                    O2t = decoder[1].propagate(O1t, n_steps=1, h0=O2t)
                    pred = softmax_layer.propagate(O2t, atype='softmax')
                    Ct   = self.get_character(pred, stype='argmax')
                    preds.append(pred)
                preds = tf.pack(preds)
            return preds


    def fp(self, X, XXM, batch_sz, num_esteps, num_dsteps):

        if num_esteps is None: num_esteps = self.TT 
        if num_dsteps is None: num_dsteps = self.TT 

        #Encoder pass
        Hts_dec, Ots_dec = [],[]
        Hts_dec_init = self.encoder_fp(X, XXM, batch_sz, num_steps=num_esteps)
        Hts_dec      = self._init_states(self.decoder, batch_sz)
        Hts_dec.pop(0)
        Hts_dec.insert(0, Hts_dec_init)

        #Decoder pass
        #C0 = last_relevant2D(X, last_indice)
        C0 = tf.constant(np.ones((tr_config['batch_sz'],), dtype='int32') * model.vocab_sz)
        pred = self.decoder_fp(C0, Hts_dec, self.decoder, \
                                            self.softmax_layer, \
                                            num_steps=num_dsteps)

        pred_skip = self.decoder_fp(C0, Hts_dec, self.skip_decoder, \
                                            self.skip_softmax_layer, \
                                            num_steps=num_dsteps)

        return pred, pred_skip


    def get_character(self, preds, stype='argmax'):
        """stype - sample type, either 'argmax' | 'multinomial' """


        #Get prediction
        #pred    = self.softmax_layer.propagate(Ht)
       
        #Sample a character
        if stype == 'multinomial':
            pred = tf.multinomial(preds, 1, seed=1234, name=None)
      
        if len(preds.get_shape()) == 2:
            sample_char = tf.argmax(preds, dimension=1)
        else:
            sample_char = tf.argmax(preds, dimension=2)
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



    def get_samples(self, X_prior, XXM, num_esteps, num_dsteps, stype='rec'):
        """Samples refers to reconstruction (only in this case)
            S           - summary variable
            num_steps   - number of total character generated 
            Hts, Ots    - is dummy variable. Needed this to synchronize 
            with sample method in char_rnn.py """

        #TODO : At test time, the prediction is currently done iteratively 
        #character by character in a greedy fashion, but eventually needs to be
        #implemented more sophisticated methods (e.g. beam search).

        preds, preds_skip  = self.fp(X_prior, XXM, X_prior.shape[0], \
                                       num_fsteps=num_esteps, \
                                       num_bsteps=num_dsteps) 

        if stype == 'skip':
            ct = self.get_character(preds_skip)
            return ct, preds_skip

        ct = self.get_character(preds)
        return ct, preds


    def _init_states(self, submodel, batch_sz):

        Hts = []

        for i, rnn in enumerate(submodel):
            if isinstance(rnn, BiGRU_LN_Layer) or \
               isinstance(rnn, GRU_LN_Layer) :  
                Hts.append(tf.zeros((batch_sz, rnn.M), dtype='float32'))
            #elif isinstance(rnn, LSTM_Content_Decoding_Layer) or \
            #     isinstance(rnn, LSTM_Content_Decoding_LN_Layer):  
            #    Hts.append(T.zeros((batch_sz, rnn.M*4), dtype=theano.config.floatX))

        return Hts
               
               
    def clone(self, new_scope_name=None):

        #TODO
        pass



