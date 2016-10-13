"""
data preprocessing

==> The model description is here: https://arxiv.org/abs/1606.01700
==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial

"""

#-------------------------------------------------------------
# modules and packages
#-------------------------------------------------------------

import os
import numpy
import cPickle as pkl


#-------------------------------------------------------------
# data pipeline
#-------------------------------------------------------------

def prepare_char_data(seqs_x, seqs_x_r, maxlen=None, n_char=205):
    """
    format input data and create auxiliary variables

    Parameters
    ----------
    seqs_x    : a list of lists of int, usually sentences in a batch
    seqs_x_r  : a list of lists of int, usually sentences in a batch, each word has been flipped
    maxlen    : int, max number of characters in a sentence
    n_char    : int, number of unique characters in the corpus 

    Returns
    -------
    x_f            : 2d numpy array, input for forward LSTM,
                     dimensions: (length of the longest sentence in a batch, batch size)
    x_r            : 2d numpy array, input for reverse LSTM,
                     dimensions: (length of the longest sentence in a batch, batch size)
    x_mask         : 2d numpy array, not in use,
                     dimensions: (length of the longest sentence in a batch, batch size)
    spaces         : 2d numpy array, binary matrix, 0 if white spaces 1 otherwise,
                     dimensions: (length of the longest sentence in a batch, batch size)
    last_chars     : 2d numpy array, binary matrix, 0 if the last char of words 1 otherwise,
                     dimensions: (length of the longest sentence in a batch, batch size)
 
    """
    assert(len(seqs_x) == len(seqs_x_r)), 'invalid inputs: length mis-match'
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x)
    x_f = numpy.zeros((maxlen_x, n_samples), dtype='int64')
    x_r = numpy.zeros((maxlen_x, n_samples), dtype='int64')
    x_mask = numpy.zeros((maxlen_x, n_samples), dtype='float32')
    spaces = numpy.zeros((maxlen_x, n_samples), dtype='float32')
    for idx, s_x in enumerate(zip(seqs_x, seqs_x_r)):
        s_x_f = numpy.asarray(s_x[0], dtype='int64')
        s_x_r = numpy.asarray(s_x[1], dtype='int64')
        s_x_f[numpy.where(s_x_f > n_char)] = 1
        s_x_r[numpy.where(s_x_r > n_char)] = 1
        x_f[:lengths_x[idx], idx] = s_x_f
        x_r[:lengths_x[idx], idx] = s_x_r
        s_x_f = numpy.pad(s_x_f, (0,spaces.shape[0]-s_x_f.shape[0]), mode='constant', constant_values=0)
        s_x_r = numpy.pad(s_x[1], (0,spaces.shape[0]-s_x_r.shape[0]), mode='constant', constant_values=0)
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        spaces[numpy.where(s_x_f != 2), idx] = 1.
        spaces[numpy.where(s_x_f == 0), idx] = 2.
    last_chars = numpy.copy(spaces)
    last_chars[0:-1,:] = last_chars[1:,:]
    last_chars[-1,:] = 2.
    spaces[numpy.where(spaces == 2.)] = 0.
    last_chars[numpy.where(last_chars == 2.)] = 1.
    for i in xrange(last_chars.shape[1]):
        last_chars[lengths_x[i] - 1,i] = 0.
    return x_f, x_r, x_mask, spaces, last_chars


def prepare_word_data(seqs_x, maxlen=None, n_words=30000):
    """
    format input data and create auxiliary variables

    Parameters
    ----------
    seqs_x  : a list of lists of int, usually sentences in a batch
    maxlen  : int, max number of characters in a sentence
    n_char  : int, number of unique characters in the corpus 

    Returns
    -------
    x_f     : 2d numpy array, input for forward LSTM,
              dimensions: (length of the longest sentence in a batch, batch size)
    x_mask  : 2d numpy array, not in use,
              dimensions: (length of the longest sentence in a batch, batch size)
 
    """
    lengths_x = [len(s) for s in seqs_x]
    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        if len(lengths_x) < 1:
            return None, None
    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    x = numpy.zeros((maxlen_x, n_samples), dtype='int64')
    x_mask = numpy.zeros((maxlen_x, n_samples), dtype='float32')
    for idx, s_x in enumerate(seqs_x):
        s_x = numpy.asarray(s_x, dtype='int64')
        s_x[numpy.where(s_x >= n_words - 1)] = 1
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx] + 1, idx] = 1.
    return x, x_mask


def load_data(path=None):
    """ 
    read a training data file (.txt)

    Parameters
    ----------
    path   : string, path to the data location

    Returns
    -------
    train  : list of string, raw text data

    """
    print '==> Loading training data'
    if not os.path.isfile(path):
        raise ValueError("The path provided from the trainset doesn't exist")
    with open(path, 'rb') as f:
        train = f.readlines()
    return train


def flip_words(s):
    """ 
    flip char order of each word for reverse LSTM, but it keeps the word order

    Example 
    -------
        If s = [7, 2, 19, 3, 42, 16, 2, 32, 20, 27, 13, 2, 10], then
        temp = [7, 2, 16, 42, 3, 19, 2, 13, 27, 20, 32, 2, 10], where "2" is a white space.

    Parameters
    ----------
    s     : list of int, single sentence, dimensions: (num of char in a sentence,)

    Returns
    -------
    temp  : list of int, char order of each word is flipped, 
            word order of a sentence is not changed, dimensions: (num of char in a sentence,)

    """
    if s[-1] is 2: 
        raise ValueError("x_r: the last char can't be a white space (2).")
    elif s[1] is not 2:
        raise ValueError("x_r: the 2nd char must be a white space (2).")
    else:
        temp = '_'.join(map(str, s)).split('_2_')
        temp = [w.split('_')[::-1] for w in temp]
        temp = '_2_'.join(['_'.join(w) for w in temp])
        temp = map(int, temp.split('_'))
        return temp


def text_to_char_index(text_data, char_dict, bos):
    """ 
    convert characters to char indices. '| ' is added to the beginning of each sentence.

    Parameters
    ----------
    text_data  : list of string, usually sentences in a batch
    char_dict  : OrderedDict, {character: index}

    Returns
    -------
               : a list of lists of int, usually sentences in a batch
    """
    return [[char_dict.get(c, 1) for c in (bos + ' ' + l.strip())] for l in text_data]


def text_to_word_index(text_data, word_dict):
    """ 
    convert words to word indices 

    Parameters
    ----------
    text_data  : list of string, usually sentences in a batch
    word_dict  : OrderedDict, {word: index}

    Returns
    -------
               : a list of lists of int, usually sentences in a batch
    """
    return [[word_dict.get(w, 1) for w in l.lower().strip().split()] for l in text_data]


def char_to_label_words(x_char, x_word):
    """ 
    create target labels

    Parameters
    ----------
    x_char       : 2d numpy array, dimensions: (length of the longest sentence in a batch, batch size)
    x_word       : a list of lists of int, sentences where words have been replaced by indices

    Returns
    -------
    label_words  : 2d numpy array, dimensions: (length of the longest sentence in a batch, batch size)

    """
    label_words = x_char.copy()
    for i in xrange(x_char.shape[1]):
        word_loc = 0
        for j in xrange(x_char.shape[0]):
            if x_char[j,i] == 2:
                word_loc += 1
                label_words[j,i] = 0
            elif x_char[j,i] == 0:
                label_words[j,i] = 0
            else:
                label_words[j,i] = x_word[word_loc, i]
    return label_words


def char_to_word_input(x_char, x_word, bos_idx):
    """ 
    create word inputs that matches with char inputs shape 

    Parameters
    ----------
    x_char          : 2d numpy array, dimensions: (length of the longest sentence in a batch, batch size)
    x_word          : list of list of int, sentences where words have been replaced by indices
    dummy_char_idx  : int, the begining of sentence symbol, we use char_dict['|']

    Returns
    -------
    word_input      : 2d numpy array, dimensions: (length of the longest sentence in a batch, batch size)

    """
    word_input = x_char.copy()
    for i in xrange(x_char.shape[1]):
        word_loc = 0
        for j in xrange(x_char.shape[0]):
            if x_char[j,i] == bos_idx and j == 0:
                word_input[j,i] = 0
                word_loc -= 1
            elif x_char[j,i] == 2:
                word_loc += 1
                word_input[j,i] = 0
            elif x_char[j,i] == 0:
                word_input[j,i] = 0
            else:
                word_input[j,i] = x_word[word_loc, i]
    return word_input


#-------------------------------------------------------------
# do all
#-------------------------------------------------------------

def txt_to_inps(x, char_dict, word_dict, opts):
    """ 
    preprocess raw text data and generate word-level and char-level inputs

    Parameters
    ----------
    x          : 2d numpy array, dimensions: (length of the longest sentence in a batch, batch size)
    char_dict  : OrderedDict, {character: index}
    word_dict  : OrderedDict, {word: index}
    opts       : dictionary, {hyperparameter: value}

    Returns
    -------
    x_f            : 2d numpy array, input for forward LSTM,
                     dimensions: (length of the longest sentence in a batch, batch size)
    x_r            : 2d numpy array, input for reverse LSTM,
                     dimensions: (length of the longest sentence in a batch, batch size)
    spaces         : 2d numpy array, binary matrix, 0 if white spaces 1 otherwise,
                     dimensions: (length of the longest sentence in a batch, batch size)
    last_chars     : 2d numpy array, binary matrix, 0 if the last char of words 1 otherwise,
                     dimensions: (length of the longest sentence in a batch, batch size)
    x_word_input_  : 2d numpy array, word-level inputs,
                     dimensions: (length of the longest sentence in a batch, batch size)
    label_words    : 2d numpy array, target labels,
                     dimensions: (length of the longest sentence in a batch, batch size)

    """
    x_f_ = numpy.asarray(text_to_char_index(x, char_dict, opts['bos']))
    x_r_ = numpy.asarray(map(flip_words, x_f_))
    x_f, x_r, _, spaces, last_chars = prepare_char_data(x_f_.tolist(), x_r_.tolist(),
                                                            maxlen=opts['maxlen'], n_char=opts['n_char'])   
    x_word_ = numpy.asarray(text_to_word_index(x, word_dict))
    x_word_, _ = prepare_word_data(x_word_, n_words=opts['n_words'])
    dummy_char_idx = char_dict[opts['bos']]
    x_word_input_ = char_to_word_input(x_f, x_word_, dummy_char_idx)
    label_words_ = char_to_label_words(x_f, x_word_)
    return x_f, x_r, spaces, last_chars, x_word_input_, label_words_


def txt_to_word_inps(x, word_dict, opts):
    """
    preprocess raw text data and generate word-level inputs
 
    Parameters
    ----------
    x             : 2d numpy array, dimensions: (length of the longest sentence in a batch, batch size)
    word_dict     : OrderedDict, {word: index}
    opts          : dictionary, {hyperparameter: value}

    Returns
    -------
    x_wordt_      : 2d numpy array, word-level inputs,
                    dimensions: (length of the longest sentence in a batch, batch size)
    x_word_mask_  : 2d numpy array, binary matrix, 1 if word 0 otherwise,
                    dimensions: (length of the longest sentence in a batch, batch size)

    """
    x_word_ = numpy.asarray(text_to_word_index(x, word_dict))
    x_word_, x_word_mask_ = prepare_word_data(x_word_, n_words=opts['n_words']) 
    return x_word_, x_word_mask_
