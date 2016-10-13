"""
Layers

==> The model description is here: https://arxiv.org/abs/1606.01700
==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial

"""

#-------------------------------------------------------------
# modules and packages
#-------------------------------------------------------------

import theano
import theano.tensor as tensor

import os
import numpy
import cPickle as pkl

from theano.ifelse import ifelse


#-------------------------------------------------------------
# utils
#-------------------------------------------------------------

def p_name(pp, name):
    """ make prefix-appended name """
    return '%s_%s' % (pp, name)

def uniform_weight(nin, nout=None):
    """ weight initilizer: uniform [-0.1, 0.1] """
    if nout is None:
        nout = nin
    W = numpy.random.rand(nin, nout) * 0.2 - 0.1
    return W.astype('float32') 

def xavier_weight(nin, nout=None):
    """ weight initilizer: Xavier Initialization """
    if nout is None:
        nout = nin
    max = numpy.sqrt(6. / (nin + nout)) 
    W = numpy.random.rand(nin, nout) * 2. * max - max
    return W.astype('float32')

def tanh(x):
    """ element-wise tanh """
    return tensor.tanh(x)

def rectifier(x):
    """ element-wise ReLU """
    return tensor.maximum(0., x)

def linear(x):
    """ element-wise linear """
    return x


#-------------------------------------------------------------
# layer initializer 
#-------------------------------------------------------------

def param_init_fullyconnected_layer(options, params, prefix, nin, nout):
    """ 
    initialize a fully connected layer
    
    Parameters
    ----------
    options  : dictionary, {hyperparameter: value}
    params   : OrderedDict, {parameter name: value}
    prefix   : string, layer name
    nin      : int, inpput dimension
    nout     : int, output dimension
    
    Returns
    -------
    params   : OrderedDict, {parameter name: value}

    """
    params[p_name(prefix, 'W')] = uniform_weight(nin, nout)
    params[p_name(prefix, 'b')] = numpy.zeros((nout,), dtype='float32')
    return params


def param_init_gate(options, params, prefix, dim):
    """ 
    initialize a gate layer

    Parameters
    ----------
    options  : dictionary, {hyperparameter: value}
    params   : OrderedDict, {parameter name: value}
    prefix   : string, layer name
    dim      : int, inpput dimension

    Returns
    -------
    params   : OrderedDict, {parameter name: value}

    """
    params[p_name(prefix, 'v')] = (numpy.random.rand(dim,) * 0.2 - 0.1).astype('float32')
    params[p_name(prefix, 'b')] = numpy.zeros((1,), dtype='float32')
    return params


def param_init_concat(options, params, prefix, nin, nout):
    """ 
    initialize a concat layer

    Parameters
    ----------
    options  : dictionary, {hyperparameter: value}
    params   : OrderedDict, {parameter name: value}
    prefix   : string, layer name
    dim      : int, inpput dimension

    Returns
    -------
    params   : OrderedDict, {parameter name: value}

    """
    params[p_name(prefix, 'W')] = uniform_weight(nin, nout)
    params[p_name(prefix, 'b')] = numpy.zeros((nout,), dtype='float32')
    return params


def param_init_lstm_uniform(options, params, prefix, nin, dim):
    """ 
    initialize a LSTM layer (language model)
    Parameters
    ----------
    options  : dictionary, {hyperparameter: value}
    params   : OrderedDict, {parameter name: value}
    prefix   : string, layer name
    nin      : int, inpput dimension
    dim      : int, inpput dimension

    Returns
    -------
    params   : OrderedDict, {parameter name: value}

    """
    params[p_name(prefix, 'W')] = numpy.concatenate([uniform_weight(nin, dim), uniform_weight(nin, dim), 
                                                     uniform_weight(nin, dim), uniform_weight(nin, dim)], axis=1)
    params[p_name(prefix, 'U')] = numpy.concatenate([uniform_weight(dim, dim), uniform_weight(dim, dim), 
                                                     uniform_weight(dim, dim), uniform_weight(dim, dim)], axis=1)
    params[p_name(prefix, 'b')] = numpy.zeros((4 * dim,), dtype='float32')
    return params


def param_init_lstm_xavier(options, params, prefix, nin, dim):
    """ 
    initialize a LSTM layer (bidirectional LSTMs)

    Parameters
    ----------
    options  : dictionary, {hyperparameter: value}
    params   : OrderedDict, {parameter name: value}
    prefix   : string, layer name
    nin      : int, inpput dimension
    dim      : int, inpput dimension

    Returns
    -------
    params   : OrderedDict, {parameter name: value}

    """
    params[p_name(prefix, 'W')] = numpy.concatenate([4. * xavier_weight(nin, dim), 4. * xavier_weight(nin, dim), 
                                                     4. * xavier_weight(nin, dim), xavier_weight(nin, dim)], axis=1)
    params[p_name(prefix, 'U')] = numpy.concatenate([4. * xavier_weight(dim, dim), 4. * xavier_weight(dim, dim), 
                                                     4. * xavier_weight(dim, dim), xavier_weight(dim, dim)], axis=1)
    params[p_name(prefix, 'b')] = numpy.zeros((4 * dim,), dtype='float32')
    return params


#-------------------------------------------------------------
# layers 
#-------------------------------------------------------------

def dropout(state_before, is_train, trng):
    """ 
    dropout with p=0.5 
        
    Parameters
    ----------
    state_before  : theano 3d tensor, input data, dimensions: (num of time steps, batch size, dim of vector)
    is_train      : theano shared scalar, 0. = test/valid, 1. = train,
    trng          : random number generator
    
    Returns
    -------
    proj          : theano 3d tensor, output data, dimensions: (num of time steps, batch size, dim of vector)
    
    """
    proj = tensor.switch(is_train, 
                         state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype), 
                         state_before * 0.5)
    return proj


def fullyconnected_layer(tparams, state_below, options, prefix, activ='lambda x: x', **kwargs):
    """ 
    compute the forward pass for a fully connected layer
    
    Parameters
    ----------
    tparams      : OrderedDict of theano shared variables, {parameter name: value}
    state_below  : theano 3d tensor, input data, dimensions: (num of time steps, batch size, dim of vector)
    options      : dictionary, {hyperparameter: value}
    prefix       : string, layer name
    activ        : string, activation function: 'liner', 'tanh', or 'rectifier'

    Returns
    -------
                 : theano 3d tensor, output data, dimensions: (num of time steps, batch size, dim of vector)

    """
    return eval(activ)(tensor.dot(state_below, tparams[p_name(prefix, 'W')]) + tparams[p_name(prefix, 'b')])


def gate_layer(tparams, X_word, X_char, options, prefix, pretrain_mode, activ='lambda x: x', **kwargs):
    """ 
    compute the forward pass for a gate layer

    Parameters
    ----------
    tparams        : OrderedDict of theano shared variables, {parameter name: value}
    X_word         : theano 3d tensor, word input, dimensions: (num of time steps, batch size, dim of vector)
    X_char         : theano 3d tensor, char input, dimensions: (num of time steps, batch size, dim of vector)
    options        : dictionary, {hyperparameter: value}
    prefix         : string, layer name
    pretrain_mode  : theano shared scalar, 0. = word only, 1. = char only, 2. = word & char
    activ          : string, activation function: 'liner', 'tanh', or 'rectifier'

    Returns
    -------
    X              : theano 3d tensor, final vector, dimensions: (num of time steps, batch size, dim of vector)

    """      
    # compute gating values, Eq.(3)
    G = tensor.nnet.sigmoid(tensor.dot(X_word, tparams[p_name(prefix, 'v')]) + tparams[p_name(prefix, 'b')][0])
    X = ifelse(tensor.le(pretrain_mode, numpy.float32(1.)),  
               ifelse(tensor.eq(pretrain_mode, numpy.float32(0.)), X_word, X_char),
               G[:, :, None] * X_char + (1. - G)[:, :, None] * X_word)   
    return eval(activ)(X)


def concat_layer(tparams, X_word, X_char, options, prefix, pretrain_mode, activ='lambda x: x', **kwargs):
    """ 
    compute the forward pass for a concat layer

    Parameters
    ----------
    tparams        : OrderedDict of theano shared variables, {parameter name: value}
    X_word         : theano 3d tensor, word input, dimensions: (num of time steps, batch size, dim of vector)
    X_char         : theano 3d tensor, char input, dimensions: (num of time steps, batch size, dim of vector)
    options        : dictionary, {hyperparameter: value}
    prefix         : string,  layer name
    pretrain_mode  : theano shared scalar, 0. = word only, 1. = char only, 2. = word & char
    activ          : string, activation function: 'liner', 'tanh', or 'rectifier'

    Returns
    -------
    X              : theano 3d tensor, final vector, dimensions: (num of time steps, batch size, dim of vector)

    """
    X = ifelse(tensor.le(pretrain_mode, numpy.float32(1.)),
               ifelse(tensor.eq(pretrain_mode, numpy.float32(0.)), X_word, X_char),
               tensor.dot(tensor.concatenate([X_word, X_char], axis=2), tparams[p_name(prefix, 'W')]) + tparams[p_name(prefix, 'b')]) 
    return eval(activ)(X)


def lstm_layer(tparams, state_below, options, prefix, spaces, **kwargs):
    """ 
    compute the forward pass for a LSTM layer (for language model)

    Parameters
    ----------
    tparams      : OrderedDict of theano shared variables, {parameter name: value}
    state_below  : theano 3d tensor, input data, dimensions: (num of time steps, batch size, dim of vector)
    options      : dictionary, {hyperparameter: value}
    prefix       : string, layer name
    spaces       : theano 2d numpy array, an element is 0 if the last char of a word, 1 otherwise

    Returns
    -------
    rval         : a tuple of theano 3d tensors, (h, c),  
                   dimensions of h/c: (num of time steps, batch size, num of hidden units)

    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    dim = tparams[p_name(prefix, 'U')].shape[0]
    init_state = tensor.alloc(0., n_samples, dim)
    init_memory = tensor.alloc(0., n_samples, dim)
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]
    def _step(s_, x_, h_, c_, U, b):
        preact = tensor.dot(h_, U)                       # Uh
        preact += x_                                     # Wx + Uh
        preact += b                                      # Wx + Uh + b
        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))  # input gate
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))  # foreget gate
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))  # output gate
        c = tensor.tanh(_slice(preact, 3, dim))          # candidate c
        c = f * c_ + i * c                               # c
        c = (1. - s_[:, None]) * c  +   s_[:, None] * c_ # update c at the last char of a word
        h = o * tensor.tanh(c)                           # h
        h = (1. - s_[:, None]) * h  +   s_[:, None] * h_ # update h at the last char of a word
        return h, c
    state_below = tensor.dot(state_below, tparams[p_name(prefix, 'W')]) + tparams[p_name(prefix, 'b')] # Wx
    rval, updates = theano.scan(_step, 
                                sequences=[spaces, state_below], 
                                outputs_info=[init_state, init_memory], 
                                non_sequences=[tparams[p_name(prefix, 'U')], tparams[p_name(prefix, 'b')]], 
                                name=p_name(prefix, '_layers'), 
                                n_steps=nsteps)
    return rval


def char_lstm_layer(tparams, state_below, options, prefix, spaces, **kwargs):
    """ 
    compute the forward pass for a LSTM layer (for bidirectional LSTMs)

    Parameters
    ----------
    tparams      : OrderedDict of theano shared variables, {parameter name: value}
    state_below  : theano 3d tensor, input data, dimensions: (num of time steps, batch size, dim of vector)
    options      : dictionary, {hyperparameter: value}
    prefix       : string, layer name
    spaces       : theano 2d numpy array, an element is 0 if white space, 1 otherwise

    Returns
    -------
    rval         : a tuple of theano 3d tensors, (h, c),
                   dimensions of h/c: (num of time steps, batch size, num of hidden units)

    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    dim = tparams[p_name(prefix, 'U')].shape[0]
    init_state = tensor.alloc(0., n_samples, dim)
    init_memory = tensor.alloc(0., n_samples, dim)
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]
    def _step(s_, x_, h_, c_, U, b):
        preact = tensor.dot(h_, U)                      # Uh
        preact += x_                                    # Wx + Uh
        preact += b                                     # Wx + Uh + b
        i = tensor.nnet.sigmoid(_slice(preact, 0, dim)) # input gate
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim)) # forget gate
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim)) # output gate
        c = tensor.tanh(_slice(preact, 3, dim))         # candidate c
        c = f * c_ + i * c                              # c
        h = o * tensor.tanh(c)                          # h    
        h = s_[:, None] * h     # reset h and c to 0 if the input char is a white space
        c = s_[:, None] * c     # s_ is a binary vector, an element is 0 if white space, 1 otherwise 
        return h, c
    state_below = tensor.dot(state_below, tparams[p_name(prefix, 'W')]) + tparams[p_name(prefix, 'b')] # Wx
    rval, updates = theano.scan(_step,
                                sequences=[spaces, state_below],
                                outputs_info=[init_state, init_memory],
                                non_sequences=[tparams[p_name(prefix, 'U')], tparams[p_name(prefix, 'b')]],
                                name=p_name(prefix, '_layers'),
                                n_steps=nsteps)
    return rval

