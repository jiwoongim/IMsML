'''The code is modified to our framework but original based from 
https://raw.githubusercontent.com/nyu-dl/dl4mt-tutorial/master/session2/data_iterator.py'''

import os, sys, numpy, gzip, collections, cPickle
import tensorflow as tf 
from utils import prepare_data
from tf_utils import last_relevant 

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def preprocess(input_file, vocab_file, chars_file):

    with open(input_file, "r") as f:
        data = f.read()

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    vocab_size = len(chars)
    vocab = dict(zip(chars, range(len(chars))))
    with open(chars_file, 'wb') as f:
        cPickle.dump(chars, f)
    with open(vocab_file, 'wb') as f:
        cPickle.dump(vocab, f)

    return vocab, chars 


class Seq2Seq_Iterator():
    """Simple Bitext iterator."""
    def __init__(self, data_dir,
                 level='char',
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1):

        source_file = os.path.join(data_dir, "sources.txt")
        target_file = os.path.join(data_dir, "targets.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        chars_file = os.path.join(data_dir, "chars.pkl")

        self.source = fopen(source_file, 'r')
        self.target = fopen(target_file, 'r')

        print("reading text file")
        self.vocab, self.chars = preprocess(source_file, vocab_file, chars_file)
        self.target_dict = self.vocab


        self.level = level
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False
        self.vocab_size  = len(self.vocab.keys())

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next_batch(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break

                if self.level == 'word':
                    self.source_buffer.append(ss.strip().split())
                    self.target_buffer.append(tt.strip().split())
                elif self.level == 'char':
                    self.source_buffer.append(list(ss.strip()))
                    self.target_buffer.append(list(tt.strip()))

            # sort by target buffer
            tlen = numpy.array([len(t) for t in self.target_buffer])
            tidx = tlen.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            _tbuf = [self.target_buffer[i] for i in tidx]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.vocab[w] if w in self.vocab else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                if len(ss) > self.maxlen and len(tt) > self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target


if __name__ == '__main__':
    data_path = '/Users/danielim/Documents/aifounded/codes/aif_ml_lib/data/aiethics/'


    abc = Seq2Seq_Iterator( data_path, \
                            level='char',\
                            batch_size=100)

    source,target = abc.next_batch()
    import pdb; pdb.set_trace()
    xx, x_mask, yy, y_mask = prepare_data(source,target, maxlen=300)
    xxx = tf.zeros([300,87,512])
    xxm = tf.Variable(x_mask.sum(1).astype('int32'))
    tmp = last_relevant(xxx, xxm)

    #TODO these source and targets are in the form of list. 
    #Based on this link below, could you put them into numpy matrix form.
    #https://danijar.com/variable-sequence-lengths-in-tensorflow/


