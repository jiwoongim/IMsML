import os, argparse, re, nltk, codecs
from tqdm import tqdm
import numpy as np

def text_to_tensor(data, max_word_length):

    max_word_length_tmp,count = 0,0
    for line in data:
        line = line.replace('<unk>', '|')
        line = line.replace('}', '')
        line = line.replace('{', '')
        for word in line.split():
            max_word_length_tmp = max(max_word_length_tmp, len(word) + 2)
            count += 1

        count += 1 # for \n

    max_word_length = min(max_word_length_tmp, max_word_length)

    char2idx = {' ':0, '{': 1, '}': 2}
    word2idx = {'<unk>': 0}
    idx2char = [' ', '{', '}']
    idx2word = ['<unk>']

    output_tensor = np.ndarray(count)
    #output_char = np.ones([count, max_word_length])
    output_char = np.ones([len(data), max_word_length])

    word_num = 0
    for ii, line in tqdm(enumerate(data)):

        line = line.lower()
        line = line.replace('<unk>', '|')
        line = line.replace('}', '')
        line = line.replace('{', '')

        #for word in line.split() + ['+']:
        #    chars = [char2idx['{']]
            #if word[0] == '|' and len(word) > 1:
            #    word = word[2:]
            #    output_tensor[word_num] = word2idx['|']
            #else:
            #    if not word2idx.has_key(word):
            #        idx2word.append(word)
            #        word2idx[word] = len(idx2word) - 1
            #    output_tensor[word_num] = word2idx[word]

        chars = [char2idx['{']]
        for char in line:
            if not char2idx.has_key(char):
                idx2char.append(char)
                char2idx[char] = len(idx2char) - 1
            chars.append(char2idx[char])
        chars.append(char2idx['}'])

        if len(chars) == max_word_length:
            chars[-1] = char2idx['}']
 
        for idx in xrange(min(len(chars), max_word_length)):
            output_char[ii, idx] = chars[idx]
        word_num += 1

    print 'Character vocabulary : %d' % len(idx2char)
    return [idx2char, char2idx], output_char
    #return [idx2word, word2idx, idx2char, char2idx], output_char
    #save(vocab_fname, [idx2word, word2idx, idx2char, char2idx])
    #save(tensor_fname, output_tensors)
    #save(char_fname, output_chars)
