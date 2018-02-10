
""" This code was from https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/utils.py """

import os, collections
from six.moves import cPickle
import numpy as np

class Seq2Seq_Loader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        source_file = os.path.join(data_dir, "sources.txt")
        target_file = os.path.join(data_dir, "targets.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        chars_file = os.path.join(data_dir, "chars.pkl")
        source_tensor_file = os.path.join(data_dir, "data_source.npy")
        target_tensor_file = os.path.join(data_dir, "data_target.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(source_tensor_file)):
            print("reading text file")
            self.source_tensor = self.preprocess(\
                    source_file, vocab_file, chars_file, source_tensor_file)
        else:
            print("loading preprocessed files")
            self.source_tensor = \
                self.load_preprocessed(chars_file, source_tensor_file)

        if not (os.path.exists(vocab_file) and os.path.exists(target_tensor_file)):
            print("reading text file")
            self.target_tensor = self.preprocess(\
                    target_file, vocab_file, chars_file, target_tensor_file)
        else:
            print("loading preprocessed files")
            self.target_tensor = \
                    self.load_preprocessed(chars_file, target_tensor_file)

        self.create_batches()
        self.reset_batch_pointer()


    def preprocess(self, input_file, vocab_file, chars_file, tensor_file):

        with open(input_file, "r") as f:
            data = f.read()

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(chars_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.vocab, f)

        tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, tensor)
        return tensor

    def load_preprocessed(self, chars_file, tensor_file):
        with open(chars_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        tensor = np.load(tensor_file)
        self.num_batches = int(tensor.size / (self.batch_size *
                                                   self.seq_length))
        return tensor

    def creat_seq2seq_batches(self):

        stopping_symbols = [self.vocab['.'], \
                            self.vocab['?'], \
                            self.vocab[';'], \
                            self.vocab[':']]

        ind_stopping_symbols = np.argwhere( np.logical_or.reduce ((
                                            self.tensor == stopping_symbols[0] , \
                                            self.tensor == stopping_symbols[1] , \
                                            self.tensor == stopping_symbols[2] , \
                                            self.tensor == stopping_symbols[3] )))

        total_batch = ind_stopping_symbols.shape[0] 

        tf.pad(tensor, paddings, mode='CONSTANT', name=None)
        pass


    def create_batches(self):

        self.num_batches = int(self.source_tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.source_tensor = self.source_tensor[:self.num_batches * self.batch_size * self.seq_length]
        self.target_tensor = self.target_tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.source_tensor
        ydata = np.copy(self.target_tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
