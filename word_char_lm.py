"""
Word-Character Recurrent Language Model 

==> The model description is here: https://arxiv.org/abs/1606.01700
==> The base code is here: https://github.com/nyu-dl/dl4mt-tutorial

"""

#-------------------------------------------------------------
# modules and packages
#-------------------------------------------------------------

import theano
import theano.tensor as tensor

import os
import copy
import yaml
import numpy
import argparse
import cPickle as pkl

from layers import uniform_weight
from layers import param_init_fullyconnected_layer, fullyconnected_layer
from layers import param_init_gate, gate_layer
from layers import param_init_concat, concat_layer
from layers import param_init_lstm_uniform, lstm_layer, char_lstm_layer
from layers import param_init_lstm_xavier

from data_preprocess import load_data, txt_to_inps 

from random import shuffle
from collections import OrderedDict
from sklearn.cross_validation import KFold
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


#-------------------------------------------------------------
# layers 
#------------------------------------------------------------- 

layers_ = {'fc': ('param_init_fullyconnected_layer', 'fullyconnected_layer'),
           'gate': ('param_init_gate', 'gate_layer'),
           'concat': ('param_init_concat', 'concat_layer'),
           'lstm_u': ('param_init_lstm_uniform', 'lstm_layer'),
           'char_lstm_x': ('param_init_lstm_xavier', 'char_lstm_layer')}


#-------------------------------------------------------------
# utils
#-------------------------------------------------------------

def zipp(params, tparams):
    """ convert parameters to Theano shared variables """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """ pull parameters from Theano shared variables """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def itemlist(tparams):
    """ get the list of parameters: Note that tparams must be OrderedDict """
    return [vv for kk, vv in tparams.iteritems()]

def init_tparams(params):
    """ initialize Theano shared variables according to the initial parameters """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def load_params(path, params):
    """ load parameters """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params

def get_layer(name):
    """ get layer initializer and layer function """
    fns = layers_[name]
    return (eval(fns[0]), eval(fns[1]))


#-------------------------------------------------------------
# initialize parameters & build model
#-------------------------------------------------------------

def init_params(options):
    """ 
    initialize all the parameters and store in OrderedDict

    Parameters
    ----------
    options  : dictionary, {hyperparameter: value}

    Returns
    -------
    params   : OrderedDict, {parameter name: value}

    """
    # store all the parameters in an OrderedDict
    params = OrderedDict()
    params['word_lookup'] = uniform_weight(options['n_words'], options['dim_word']) 
    params['char_lookup_f'] = uniform_weight(options['n_char'], options['dim_char'])
    params['char_lookup_r'] = uniform_weight(options['n_char'], options['dim_char'])
    params = get_layer('char_lstm_x')[0](options, params, 'bi_lstm_f', options['dim_char'], options['dim_bi_lstm'])    
    params = get_layer('char_lstm_x')[0](options, params, 'bi_lstm_r', options['dim_char'], options['dim_bi_lstm'])
    params = get_layer('fc')[0](options, params, 'combine_hs', options['dim_bi_lstm'] * 2, options['dim_word'])
    if options['model_type'] == 'gate':
        params = get_layer('gate')[0](options, params, 'gate', options['dim_word'])
    else:
        params = get_layer('concat')[0](options, params, 'concat', options['dim_word'] * 2, options['dim_word'])
    params = get_layer('lstm_u')[0](options, params, 'lstmlm_1', options['dim_word'], options['dim_lm_lstm'])
    params = get_layer('lstm_u')[0](options, params, 'lstmlm_2', options['dim_lm_lstm'], options['dim_lm_lstm'])
    params = get_layer('fc')[0](options, params, 'pre_softmax', options['dim_lm_lstm'], options['n_words'])
    return params


def build_model(tparams, options):
    """ 
    build training procedure

    Parameters
    ----------
    tparams  : OrderedDict of theano shared variables, {parameter name: value}
    options  : dictionary, {hyperparameter: value}

    Returns
    -------
    trng           : random number generator
    is_train       : theano shared scalar, flag for train(1.) or test(0.)
    pretrain_mode  : theano shared scalar, flag for pretraining mode: word(0.), char(1.), or both(2.)
    x_f            : theano 2d tensor, char input for forward LSTM
    x_r            : theano 2d tensor, char input for reverse LSTM
    x_spaces,      : theano 2d tensor, binary matrix, 0 if white spaces 1 otherwise
    x_last_chars   : theano 2d tensor, binary matrix, 0 if the last char of words 1 otherwise
    x_word_input   : theano 2d tensor, word-level inputs
    label_words    : theano 2d tensor, target labels
    cost           : theano tensor scalar, symbolic computation of the forward pass

    """
    # declare theano variables
    trng = RandomStreams(1234)                                          
    is_train = theano.shared(numpy.float32(0.))                         
    pretrain_mode = theano.shared(numpy.float32(0.))                    
    x_f = tensor.matrix('x_f', dtype='int64')
    x_r = tensor.matrix('x_r', dtype='int64')
    x_spaces = tensor.matrix('spaces', dtype='float32')                 
    x_last_chars = tensor.matrix('last_chars', dtype='float32') 
    x_word_input = tensor.matrix('x_word_in', dtype='int64')           
    label_words = tensor.matrix('label_words', dtype='int64')          
    n_timesteps = x_f.shape[0]  
    n_samples = x_f.shape[1]

    # word-based embeddings
    Wemb = tparams['word_lookup'][x_word_input.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # char-based embeddings / bidirectional LSTM
    emb_c = tparams['char_lookup_f'][x_f.flatten()].reshape([n_timesteps, n_samples, options['dim_char']]) 
    proj_f = get_layer('char_lstm_x')[1](tparams, emb_c, options, 'bi_lstm_f', x_spaces)
    emb_c = tparams['char_lookup_r'][x_r.flatten()].reshape([n_timesteps, n_samples, options['dim_char']])   
    proj_r = get_layer('char_lstm_x')[1](tparams, emb_c, options, 'bi_lstm_r', x_spaces)
    proj = tensor.concatenate([proj_f[0], proj_r[0]], axis=2)
    Cemb = get_layer('fc')[1](tparams, proj, options, 'combine_hs', activ='linear')

    # mix word-based and char-based embeddings by gate or concat layer
    if options['model_type'] == 'gate':
        proj = get_layer('gate')[1](tparams, Wemb, Cemb, options, 'gate', pretrain_mode, activ='linear')
    else:
        proj = get_layer('concat')[1](tparams, Wemb, Cemb, options, 'concat', pretrain_mode, activ='linear')

    # 2-layer LSTM
    proj = get_layer('lstm_u')[1](tparams, proj, options, 'lstmlm_1', x_last_chars)
    proj = get_layer('lstm_u')[1](tparams, proj[0], options, 'lstmlm_2', x_last_chars)
    proj_h = proj[0] 

    # dropout
    if options['use_dropout']:
        proj_h = dropout(proj_h, is_train, trng)

    # softmax
    logit = get_layer('fc')[1](tparams, proj_h, options, 'pre_softmax', activ='linear')        
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))
   
    # mask for final loss: 1 if the last char of a word, 0 otherwise
    final_loss_mask = 1 - x_last_chars                
    final_loss_mask = final_loss_mask.flatten()
    
    # cost
    x_flat = label_words.flatten() 
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx] + 1e-8) * final_loss_mask # only last chars of words 
    cost = cost.reshape([x_f.shape[0], x_f.shape[1]]) # reshape to n_steps x n_samples
    cost = cost.sum(0)                                # sum up NLL of words in a sentence
    cost = cost.mean()                                # take mean of sentences

    return trng, is_train, pretrain_mode, x_f, x_r, x_spaces, x_last_chars, x_word_input, label_words, cost


#-------------------------------------------------------------
# perplexity
#-------------------------------------------------------------

def perplexity(f_cost, lines, char_dict, word_dict, opts):
    """ 
    compute perplexity over the validation/test data

    Parameters
    ----------
    f_cost     : compiled function, computation for the forward pass
    lines      : list of string, validation/test data
    char_dict  : OrderedDict, {character: index}
    word_dict  : OrderedDict, {word: index}
    opts       : dictionary, {hyperparameter: value}

    Returns
    -------
    cost       : numpy float32, perplexity 

    """
    n_lines = len(lines)
    cost = 0.
    n_words = 0.
    total_n_words = 0.
    batch_size = 64  
    kf_train = KFold(n_lines, n_folds=n_lines/(batch_size-1), shuffle=False)
    for _, index in kf_train:
        x = [lines[i] for i in index]
        x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_ \
                                                      = txt_to_inps(x, char_dict, word_dict, opts)
        n_words = (1 - x_last_chars_).sum()
        cost_one = f_cost(x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_) * x_f_.shape[1]
        cost += cost_one
        total_n_words += n_words  
    cost = numpy.exp(cost / total_n_words)
    return cost


#-------------------------------------------------------------
# optimizer
#-------------------------------------------------------------

def sgd(lr, tparams, grads, inp, cost):
    """ 
    build stochastic gradient descent

    Parameters
    ----------
    lr             : theano tensor scalar,
    tparams        : OrderedDict of theano shared variables, {parameter name: value}
    grads          : theano symbolic gradients, 
    inp            : list of tensors, input data and auxiliary variables
    cost           : theano tensor scalar, symbolic computation of the forward pass

    Returns
    -------
    f_grad_shared  : compiled function, compute gradients
    f_update       : compiled function, update parameters

    """
    profile = False
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)
    paramup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=paramup, profile=profile)
    return f_grad_shared, f_update


#-------------------------------------------------------------
# training
#-------------------------------------------------------------

def train(opts):
    """ training process starts here """
    
    print '==> Training a language model'  
    if opts['model_type'] == 'gate':
        if opts['pretrain'] > 0:
            print '    [Gated Word & Char with Pretraining]'
        else:
            print '    [Gated Word & Char]'
    else:
        if opts['pretrain'] > 0:
            print '    [Concat Word & Char with Pretraining]'
        else:
            print '    [Concat Word & Char]'

    #---------------------------------------------------------
    # prepare ingredients
    #---------------------------------------------------------   

    print '==> Loading dictionaries: ',
    
    # load word dictionary
    print 'word dict,',
    if opts['word_dictionary']:
        with open(opts['word_dictionary'], 'rb') as f:
            word_dict = pkl.load(f) # word -> index 
        word_idict = dict()
        for kk, vv in word_dict.iteritems():
            word_idict[vv] = kk     # index -> word
    
    # load character dictionary
    print 'char dict,',
    if opts['char_dictionary']:
        with open(opts['char_dictionary'], 'rb') as f:
            char_dict = pkl.load(f) # char -> index
            char_dict[opts['bos']] = len(char_dict) # add the BOS symbol 
        char_idict = dict()
        for kk, vv in char_dict.iteritems():
            char_idict[vv] = kk     # index -> char
    print 'Done'        
    
    # reload options 
    if opts['reload_'] and os.path.exists(opts['saveto']):
        with open('%s.pkl' % opts['saveto'], 'rb') as f:
            reloaded_options = pkl.load(f)
            opts.update(reloaded_options)
   
    # load training data
    train = load_data(path=opts['train_text'])
 
    # initialize params
    print '==> Building model:'
    params = init_params(opts)

    # reload parameters
    if opts['reload_'] and os.path.exists(opts['saveto']):
        params = load_params(opts['saveto'], params)

    # convert params to Theano shared variabel 
    tparams = init_tparams(params)
    
    # build computational graph 
    trng, is_train, pretrain_mode, x_f, x_r, x_spaces, x_last_chars, x_word_input, label_words, cost \
                                                                              = build_model(tparams, opts)
    inps = [x_f, x_r, x_spaces, x_last_chars, x_word_input, label_words]

    print '==> Building f_cost...',
    f_cost = theano.function(inps, cost)
    print 'Done'

    # get gradients
    print '==> Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))

    # gradient clipping
    print 'gradient clipping...',
    grad_norm = tensor.sqrt(tensor.sum([tensor.sum(g**2.) for g in grads]))
    tau = opts['gradclip']
    grad_clipped = []
    for g in grads:
        grad_clipped.append(tensor.switch(tensor.ge(grad_norm, tau), g * tau / grad_norm, g))
    print 'Done'

    # build optimizer
    lr = tensor.scalar(name='lr')
    print '==> Building optimizers...',
    f_grad_shared, f_update = eval(opts['optimizer'])(lr, tparams, grad_clipped, inps, cost)
    print 'Done'
 
    #---------------------------------------------------------
    # start optimization
    #---------------------------------------------------------   

    print '==> Optimization:'

    # reload history
    history_errs = []
    if opts['reload_'] and os.path.exists(opts['saveto']):
        history_errs = list(numpy.load(opts['saveto'])['history_errs'])
    best_p = None
    bad_counter = 0

    # load validation and test data
    if opts['valid_text']:
        valid_lines = []
        with open(opts['valid_text'], 'r') as f:
            for l in f:
                valid_lines.append(l)
        n_valid_lines = len(valid_lines)
    if opts['test_text']:
        test_lines = []
        with open(opts['test_text'], 'r') as f:
            for l in f:
                test_lines.append(l)
        n_test_lines = len(test_lines)
    
    # initialize some values
    uidx = 0                 # update counter
    estop = False            # early stopping flag
    m = opts['pretrain']     # pretrain for m epochs using word/char only
    lrate = opts['lrate']
    lr_decayed = opts['lrate']
    batch_size = opts['batch_size']

    # outer loop: epochs
    for eidx in xrange(opts['max_epochs']):
        
        n_samples = 0  # sample counter
              
        # shuffle training data every epoch
        print '==> Shuffling sentences...',
        shuffle(train)
        print 'Done'
      
        # learning rate decay
        if eidx >= opts['lr_decay_start']:
            lr_decayed /= opts['lr_decay'] 

        # set pretraining mode
        if eidx in [e for e in range(m)]:
            pretrain_mode.set_value(0.)
        elif eidx in [e + m for e in range(m)]:
            lrate = .1
            pretrain_mode.set_value(1.)
        else:
            lrate = lr_decayed
            pretrain_mode.set_value(2.)      
        print 'pretrain_mode = ', pretrain_mode.get_value(), 'epoch = ', eidx, 'lr = ', lrate
 
        # training iterator 
        kf_train = KFold(len(train), n_folds=len(train)/(batch_size-1), shuffle=False)
  
        # inner loop: batches
        for _, index in kf_train:
            n_samples += len(index)
            uidx += 1

            # is_train=1 at training time
            is_train.set_value(1.)

            # get a batch
            x = [train[i] for i in index]
                
            # format input data
            x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_ \
                                                      = txt_to_inps(x, char_dict, word_dict, opts) 
            
            # compute cost 
            cost = f_grad_shared(x_f_, x_r_, x_spaces_, x_last_chars_, x_word_input_, label_words_)     

            # update parameters 
            f_update(lrate)

            # check cost  
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.
  
            # display cost
            if numpy.mod(uidx, opts['dispFreq']) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

            # save params
            if numpy.mod(uidx, opts['saveFreq']) == 0:
                print 'Saving...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(opts['saveto'], history_errs=history_errs, **params)
                pkl.dump(opts, open('%s.pkl' % opts['saveto'], 'wb'))
                print 'Done'

            # compute validation/test perplexity
            if numpy.mod(uidx, opts['validFreq']) == 0:
                print "Computing Dev/Test Perplexity"
                
                # is_train=0 at valid/test time
                is_train.set_value(0.)                  
                valid_err = perplexity(f_cost, valid_lines, char_dict, word_dict, opts)               
                test_err = perplexity(f_cost, test_lines, char_dict, word_dict, opts)
                history_errs.append([valid_err, test_err])
                
                # save the best params
                if len(history_errs) > 1:
                    if uidx == 0 or valid_err <= numpy.array(
                            history_errs)[:, 0].min():
                        best_p = unzip(tparams)
                        print 'Saving best params...',
                        numpy.savez(opts['savebestto'], history_errs=history_errs, **params)
                        pkl.dump(opts, open('%s.pkl' % opts['savebestto'], 'wb'))
                        print 'Done'
                        bad_counter = 0
                    if len(history_errs) > opts['patience'] and valid_err >= numpy.array(
                                history_errs)[:-opts['patience'], 0].min():
                        bad_counter += 1
                        if bad_counter > opts['patience']:
                            print 'Early Stop!'
                            estop = True
                            break

                print 'Valid ', valid_err, 'Test ', test_err 
   
        # inner loop: end
  
        print 'Seen %d samples' % n_samples

        # early stopping
        if estop:
            break
    
    # outer loop: end 
   
    if best_p is not None:
        zipp(best_p, tparams)
    
    # compute validation/test perplexity at the end of training
    is_train.set_value(0.)
    valid_err = perplexity(f_cost, valid_lines, char_dict, word_dict, opts)
    test_err = perplexity(f_cost, test_lines, char_dict, word_dict, opts)
    print 'Valid ', valid_err, 'Test ', test_err

    # save everithing
    params = copy.copy(best_p)
    numpy.savez(opts['saveto'], zipped_params=best_p, valid_err=valid_err, 
                test_err=test_err, history_errs=history_errs, **params)

    return valid_err, test_err


#-------------------------------------------------------------
# run
#-------------------------------------------------------------

if __name__ == '__main__':

    # load model configs
    parser = argparse.ArgumentParser()
    parser.add_argument("yml_location", help="Location of the yml file", type=argparse.FileType('r')) 
    args = parser.parse_args()
    options = yaml.load(args.yml_location)
    
    # train a language model
    best_valid, best_test = train(options)
