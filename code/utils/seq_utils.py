import os, sys, math, time, inspect
import numpy as np


def encode_seq(script, dic):
    """Returns decription of the script (sequence) based on dictionary dic"""

    decription = []
    ##  Convert text to indice
    for char in script:

        if isinstance(char, type(np.asarray([0]))) : char = char[0]
        ind = np.asarray([dic[char]])
        decription.append(ind)
    decription = np.asarray(decription, dtype='int32').T
    return decription


def decode_seq(script, dic):
    """Returns decription of the script (sequence) based on dictionary dic"""

    decription = ""
    ##  Convert text to indice
    for ind in script:

        #if isinstance(char, type(np.asarray([0]))) : char = char[0]
        char = dic[ind]
        decription += char

    return decription



'''This code was from 
https://github.com/nyu-dl/dl4mt-tutorial/blob/master/session2/nmt.py'''
# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1 #This will give tight bound
    maxlen_y = np.max(lengths_y) + 1 #This will give tight bound

    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    y = np.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return  x.T, x_mask.T, y.T, y_mask.T


