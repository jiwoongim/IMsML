import os, time, random
import numpy as np
import tensorflow as tf

from collections import deque
from datail utils import create_batch

from utils.utils import decode_seq, prepare_data
from utils.tf_utils import last_relevant 


class Trainer(object):
    

    def __init__(self, tr_config, model_config):

        self.tr_config = tr_config
        self.model_config = model_config


    def compile(self, model, optimizer):

        self.inputs     = tf.placeholder(tf.int32, [None, None])
        self.targets    = tf.placeholder(tf.int32, [None, None])
        self.batch_sz   = tf.placeholder(tf.int32)

        cost    = model.cost(self.inputs, self.targets, \
                            self.input_masks, self.target_masks)
        optim   = optimizer.minimize(cost)
        fevals  = [cost, optim]

        return fevals


    def train(self, session, fevals, model, data_loader):

        #Saver
        saver = tf.train.Saver(tf.all_variables())

        for epoch in xrange(self.tr_config['num_updates']): 

            start_time = time.time()     
            source_i, target_i = data_loader.next_batch()
            XX, YY = create_batch(self.tr_config['vocab'], \
                                  self.tr_config['data_dir'], \
                                  self.tr_config['num_steps'])

            cost, _ = session.run(fevals, {\
                    self.inputs  : XX,\
                    self.targets : YY,\
                    self.batch_sz: XX.shape[0])
            end_time = time.time()
            epoch_time = end_time - start_time
            print("...Epoch %d, Train Cost %f, Time %f" % \
                                (epoch, cost, epoch_time))

            if epoch % 1000 == 0 :
                if self.tr_config['summaryF'] and model.summarize is not None:
                    summary_str = session.run(model.summarize, {\
                                        self.inputs  : XX,\
                                        self.targets : YY})
                    model.summary_writer.add_summary(summary_str, epoch)

            prior="The future of artificial intelligence "
            ## Getting sample takes up (relative to GPU) a lot of time in CPU 
            if self.tr_config['sampleF'] and epoch % 200 == 0 : 

                ## Gen samples
                gen_txt = self.get_samples(session, \
                                             model, \
                                             data_loader,\
                                             prior,\
                                             num_steps=100) 

                ## Save the samples to a log file
                gen_text_epoch = '-'*10+'Epoch '+str(epoch)+'-'*10 + '\n' +\
                                prior + gen_txt + '\n\n'

                f = open(self.tr_config['save_gen_text'], 'a')
                f.write(gen_text_epoch)
                f.close()


            ## TODO Checkpoint (not working)
            #if epoch % 500 ==  (500 - 1) :

            #    checkpoint_path = self.tr_config['checkpoint_path']
            #    saver.save(session, checkpoint_path, global_step \
            #                    = epoch * data_loader.num_batches)
            #    print("model saved to {}".format(checkpoint_path))


        return cost


    def get_samples(self, session, model, data_loader, prior='First ', num_steps=1000):
        ''' Continue generate text startin from prior text, where
                    prior="The "  '''

        ind_prior = []
        prior_len = len(prior)
        ind2voc = data_loader.chars
        voc2ind = data_loader.vocab
        self.prior_inputs = tf.placeholder(tf.int32)

        ##  Convert text to indice
        ind_prior = decode_seq(prior, voc2ind)

        ## Using RNN to obtain the context vector "last hidden vector" 
        Ct, Hts, Ots = model.get_context(self.prior_inputs, num_steps=prior_len)
        #S           = tf.reshape(S, [1]) #Grab last symbol

        ## Continous generating text starting from context vector and last symbol
        fgen_indice = model.sample(Ct, num_steps, Hts=Hts, Ots=Ots)
        gen_indice  = session.run(fgen_indice, { self.prior_inputs: ind_prior })

        ##  Convert indice to text
        text = decode_seq(gen_indice, ind2voc)[0]
        return ''.join(text)


